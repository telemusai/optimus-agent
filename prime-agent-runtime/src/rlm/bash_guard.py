"""Bounded accident prevention for literal shell commands, not a security sandbox.

See docs/bash-accident-guards.md for the supported subset and deliberate limits.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import selectors
import time
from pathlib import Path

MAX_COMMAND = 65536
MAX_TOKENS = 4096
MAX_PROBES = 8
PROBE_SECONDS = 2.0
PROBE_BYTES = 65536


class DestructiveGitRefusalError(RuntimeError):
    pass


class DestructiveRmRefusalError(RuntimeError):
    pass


class BashGuardRefusalError(RuntimeError):
    pass


def _refuse(kind: str, reason: str) -> None:
    if kind == "git":
        raise DestructiveGitRefusalError(
            "bash(): refused a destructive git command. " + reason +
            " Commit, stash, or back up your work first (staging alone is not enough). "
            "After intentional review, use allow_destructive_git=True for this call."
        )
    raise DestructiveRmRefusalError(
        "bash(): refused recursive-force rm. " + reason +
        " Review the target paths; after intentional review, use "
        "allow_destructive_rm=True for this call."
    )


def _literal(word: str) -> bool:
    # Quote removal deliberately does not make expansions look trustworthy.
    return bool(word) and not any(c in word for c in "$`*?[]{}~\\")


def _basename(word: str) -> str:
    return word.rsplit("/", 1)[-1]


def _without_comments(command: str) -> str:
    # Keep newlines as command boundaries; shlex's own comment reader consumes them.
    result: list[str] = []
    quote = ""
    boundary = True
    index = 0
    while index < len(command):
        char = command[index]
        if char == chr(92) and quote != "'" and index + 1 < len(command):
            result.extend(command[index:index + 2])
            index += 2
            boundary = False
            continue
        if not quote and char == "#" and boundary:
            end = command.find("\n", index)
            index = len(command) if end < 0 else end
            continue
        if char in ("'", '"'):
            if not quote:
                quote = char
            elif quote == char:
                quote = ""
        result.append(char)
        boundary = not quote and (char.isspace() or char in ";&|()<>")
        index += 1
    return "".join(result)


def _segments(command: str) -> list[tuple[list[str], str]]:
    # Splice exactly as the shell does, including inside a command word.
    lexer = shlex.shlex(_without_comments(command.replace(chr(92) + "\n", "")),
                        posix=True, punctuation_chars=";&|()<>\n")
    lexer.commenters = ""
    lexer.whitespace = " \t\r"
    lexer.whitespace_split = True
    words: list[str] = []
    result: list[tuple[list[str], str]] = []
    for count, word in enumerate(lexer):
        if count >= MAX_TOKENS:
            raise ValueError("token limit")
        if word and all(c in ";&|()<>\n" for c in word):
            separator = ";" if all(c in ";\n" for c in word) else word
            result.append((words, separator))
            words = []
        else:
            words.append(word)
    result.append((words, ""))
    return result


def _unwrap(words: list[str]) -> tuple[list[str], bool]:
    """Recognize common literal wrappers; never execute their environment changes."""
    uncertain = False
    index = 0
    while index < len(words):
        word = words[index]
        if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*=.*", word):
            uncertain = True
            index += 1
        elif _basename(word) in ("command", "exec", "sudo", "env"):
            wrapper = _basename(word)
            uncertain |= wrapper in ("sudo", "env")
            index += 1
            while index < len(words) and words[index].startswith("-"):
                option = words[index]
                index += 1
                if wrapper in ("sudo", "env"):
                    uncertain = True
                    takes_value = ({"-u", "-g", "-h", "-p", "-C", "-T", "-R", "-D",
                                    "--user", "--group", "--host", "--prompt", "--chdir", "--chroot",
                                    "--close-from", "--command-timeout", "--role", "--type"}
                                   if wrapper == "sudo" else {"-C", "--chdir", "-u", "--unset"})
                    if option in takes_value and index < len(words):
                        index += 1
                if option == "--":
                    break
        else:
            break
    return words[index:], uncertain


def _git_site(words: list[str], cwd: Path) -> tuple[Path, bool, bool] | None:
    """Return target, ignored-file mode and unresolved-global-option flag."""
    args = words[1:]
    index = 0
    unknown = False
    while index < len(args) and args[index] not in ("checkout", "restore", "reset", "clean"):
        option = args[index]
        if option == "-C" and index + 1 < len(args):
            path = args[index + 1]
            if _literal(path):
                cwd = cwd / path
            else:
                unknown = True
            index += 2
        elif option.startswith("-C") and len(option) > 2:
            path = option[2:]
            if _literal(path):
                cwd = cwd / path
            else:
                unknown = True
            index += 1
        elif option in ("--no-pager", "--literal-pathspecs", "--no-optional-locks"):
            index += 1
        elif option in ("-c", "--git-dir", "--work-tree", "--namespace", "--config-env"):
            unknown = True
            index += 2
        elif option.startswith("-"):
            unknown = True
            index += 1
        else:
            return None  # A different subcommand, not text in its arguments.
    if index >= len(args):
        return None
    subcommand = args[index]
    tail = args[index + 1:]
    options = tail[:tail.index("--")] if "--" in tail else tail
    flags = [word for word in options if word.startswith("-")]
    short = "".join(word[1:] for word in flags if not word.startswith("--"))
    if subcommand == "reset":
        destructive = "--hard" in flags
    elif subcommand == "clean":
        destructive = ("f" in short or "--force" in flags) and not ("n" in short or "--dry-run" in flags)
    elif subcommand == "restore":
        destructive = bool(tail) and not ("--help" in flags or "h" in short)
    else:
        # Include named pathspecs, not only the upstream whole-tree '.' idiom.
        destructive = ("--" in tail or any(flag in flags for flag in ("--force", "--ours", "--theirs", "--merge"))
                       or any(c in short for c in "fm") or any(flag.startswith("--conflict") for flag in flags))
        # A tree-ish followed by a path can discard without '--'.
        destructive |= len([word for word in options if not word.startswith("-")]) >= 2
    if not destructive:
        return None
    return cwd, subcommand == "clean" and ("x" in short or "X" in short), unknown


def _rm_operands(words: list[str]) -> list[str] | None:
    recursive = force = False
    operands: list[str] = []
    options = True
    unknown = False
    for word in words[1:]:
        if options and word == "--":
            options = False
        elif options and word.startswith("--"):
            recursive |= word == "--recursive"
            force |= word == "--force"
            unknown |= word not in ("--recursive", "--force", "--verbose", "--one-file-system", "--preserve-root", "--preserve-root=all", "--no-preserve-root")
        elif options and word.startswith("-") and word != "-":
            recursive |= "r" in word[1:] or "R" in word[1:]
            force |= "f" in word[1:]
            unknown |= any(c not in "rRfivI" for c in word[1:])
        else:
            operands.append(word)
    if not (recursive and force):
        return None
    return [] if unknown else operands


def _probe(cwd: Path, ignored: bool, env: dict[str, str], deadline: float) -> list[str]:
    """Direct read-only argv probe. No shell replay, pipe readers, or unbounded wait."""
    if os.name != "posix":
        _refuse("git", "A bounded native status probe is unavailable on this platform.")
    if any(key.startswith("GIT_") for key in env):
        _refuse("git", "Git environment overrides make the repository target uncertain.")
    git = shutil.which("git", path=env.get("PATH", os.defpath))
    if not git:
        _refuse("git", "Git status is unavailable.")
    argv = [git, "--no-optional-locks", "-c", "core.fsmonitor=false", "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    if ignored:
        argv.append("--ignored=matching")
    proc = None
    reaped = False
    try:
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                start_new_session=True, bufsize=0)
        assert proc.stdout is not None
        data = bytearray()
        # No reader thread: closing the pipe never waits for a descendant retaining it.
        os.set_blocking(proc.stdout.fileno(), False)
        with selectors.DefaultSelector() as selector:
            selector.register(proc.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    _refuse("git", "Git status exceeded its time limit.")
                if not selector.select(min(remaining, 0.05)):
                    continue
                chunk = os.read(proc.stdout.fileno(), min(8192, PROBE_BYTES + 1 - len(data)))
                if not chunk:
                    break
                data.extend(chunk)
                if len(data) > PROBE_BYTES:
                    _refuse("git", "Git status exceeded its output limit.")
        try:
            status = proc.wait(timeout=max(0, deadline - time.monotonic()))
            reaped = True
        except subprocess.TimeoutExpired:
            _refuse("git", "Git status exceeded its time limit.")
        if status != 0:
            _refuse("git", "Git status could not verify this repository.")
        return [part.decode("utf-8", "replace") for part in data.split(b"\0") if part]
    except OSError:
        _refuse("git", "Git status could not be started or read.")
    finally:
        if proc is not None:
            # Never signal a recycled PGID after wait() released the leader identity.
            # The trusted status command has no expected background descendants.
            if not reaped:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except OSError:
                    pass
            if proc.stdout is not None:
                proc.stdout.close()
            if not reaped:
                try:
                    proc.wait(timeout=0.2)
                except (OSError, subprocess.TimeoutExpired):
                    pass
    return []


def guard_command(command: str, *, cwd: str, env: dict[str, str],
                  allow_destructive_git: bool = False, allow_destructive_rm: bool = False) -> None:
    """Check literal supported invocations before any BashHandle is constructed."""
    if type(allow_destructive_git) is not bool or type(allow_destructive_rm) is not bool:
        raise TypeError("destructive-command overrides must be bool")
    if allow_destructive_git and allow_destructive_rm:
        return
    if len(command) > MAX_COMMAND:
        raise BashGuardRefusalError("bash(): command exceeds accident-guard scan limit; split it into smaller calls.")
    try:
        segments = _segments(command)
    except ValueError:
        raise BashGuardRefusalError("bash(): command cannot be checked within the bounded shell syntax; simplify it.") from None
    root = Path(cwd).resolve()
    target = root
    uncertain = False
    probes = 0
    prior_effects = False
    deadline = time.monotonic() + PROBE_SECONDS
    for raw, separator in segments:
        words, wrapped = _unwrap(raw)
        if not words:
            uncertain |= wrapped or separator not in ("", ";", "\n", "&&", "||")
            continue
        name = _basename(words[0])
        if name == "cd":
            if (len(words) == 2 and _literal(words[1]) and ".." not in Path(words[1]).parts
                    and (not env.get("CDPATH") or Path(words[1]).is_absolute())
                    and separator == "&&" and not wrapped):
                target = target / words[1]
            else:
                uncertain = True
        elif name in ("pushd", "popd", "eval", "source", ".", "export", "unset", "alias", "function"):
            uncertain = True
        elif name == "git" and not allow_destructive_git:
            site = _git_site(words, target)
            if site:
                repo, ignored, unknown = site
                if prior_effects:
                    _refuse("git", "Earlier commands may change the worktree; use a separate call after inspecting their results.")
                if uncertain or wrapped or unknown or separator not in ("", ";", "\n", "&&", "||"):
                    _refuse("git", "Shell syntax or repository relocation is not statically supported.")
                probes += 1
                if probes > MAX_PROBES or time.monotonic() >= deadline:
                    _refuse("git", "The per-call status-probe budget was exhausted.")
                dirty = _probe(repo, ignored, env, deadline)
                if dirty:
                    paths = "; ".join(repr(line[:160]) for line in dirty[:10])
                    extra = " (more paths omitted)" if len(dirty) > 10 else ""
                    _refuse("git", "Uncommitted paths: " + paths + extra + ".")
        elif name == "rm" and not allow_destructive_rm:
            operands = _rm_operands(words)
            if operands is not None:
                if os.name != "posix":
                    _refuse("rm", "Native shell path translation is not supported on this platform.")
                if uncertain or wrapped or separator not in ("", ";", "\n", "&&", "||"):
                    _refuse("rm", "Shell syntax or directory relocation is not statically supported.")
                if not operands:
                    _refuse("rm", "No explicit, supported operand list was found.")
                for operand in operands:
                    if not _literal(operand) or operand == "-":
                        _refuse("rm", "An operand is dynamic or unsupported.")
                    path = Path(operand)
                    if any(part == ".." or (part.startswith(".") and part != ".") for part in path.parts):
                        _refuse("rm", "An operand contains a parent or hidden path component.")
                    try:
                        resolved = (target / path).resolve()
                        home = Path(env["HOME"]).resolve() if env.get("HOME") else None
                        inside = (resolved.is_relative_to(root) and resolved != root
                                  and resolved != home and root != Path(root.anchor))
                    except (OSError, RuntimeError, ValueError):
                        inside = False
                    if not inside:
                        _refuse("rm", "An operand resolves outside the working directory or to its root.")
        prior_effects |= name != "cd"
        # Groups, pipelines and redirections make subsequent shell state opaque.
        uncertain |= separator not in ("", ";", "\n", "&&", "||")
