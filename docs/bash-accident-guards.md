# Kernel Bash accident guards

`bash(command)` keeps its existing process ownership, asynchronous handle,
cancellation, prefix, environment, and output behavior. Before constructing a
`BashHandle`, the runtime now checks a bounded subset of literal shell commands.
These checks prevent common accidents. **They are not a security sandbox or an
authorization boundary.** Python and the shell still have their existing powers.

## Covered forms

- Literal `git reset --hard`, `git restore` with arguments, forced `git clean`
  (except dry runs), and discard forms of `git checkout`: `--` pathspecs,
  force/ours/theirs/merge/conflict options, or tree-ish plus path.
- Literal `rm` with both recursive and force flags, including short combined
  flags, long flags, flags after operands, and `--` option termination.
- Single/double-quoted or escaped literal command names and slash-qualified
  tool names. Quoted text passed to `echo` or `printf` is not a command.
- Direct sequences separated by semicolon, newline, `&&`, or `||`. A detected
  destructive Git command after any non-`cd` command refuses: earlier commands
  could create work after the preflight (`touch new; git clean -f`, for example).
  Use separate calls and inspect the earlier result. Static
  `cd path && ...` and Git `-C` options address their target directory. A `cd`
  with a parent component, a different separator, or a dynamic target makes
  later detected destructive commands refuse rather than probe the wrong tree.
  Relative `cd` also refuses when `CDPATH` is nonempty, since the shell may select
  a different directory. An absolute `cd` target does not use `CDPATH`.
- Simple `command`/`exec` wrappers. Detected destructive commands with inline
  assignments (including assignment-only prefix commands), `env`, `sudo`, unknown Git globals, redirections, groups or
  uncertain directory state refuse. Git environment overrides also refuse.
  The command prefix is scanned as part of the same text but never executed
  by the status probe.

The Git guard checks the whole addressed repository, not just a requested
pathspec. Staged, unstaged, and untracked work triggers a refusal. `clean -x`
and `clean -X` also check ignored files. Messages show up to ten truncated,
escaped porcelain records. Commit, stash, or back up work; **staging alone does
not protect work from `reset --hard`**. Probe failures and non-repositories
refuse detected destructive commands rather than claiming the target is clean.

The rm workspace boundary is the kernel's current working directory at the
start of this call, resolved through symlinks. A literal operand must resolve
strictly inside that directory. Using a filesystem root as the workspace refuses
recursive-force rm. The directory itself, HOME itself, parent components,
hidden components such as `.git` and `.env`, expansions/globs, stdin-style
operands, and missing operand lists refuse. A nonexistent literal path inside
the boundary is permitted (normal `rm -f` behavior). Kernel `os.chdir()` changes
the boundary for later calls; it is not a pinned project access policy.

## Intentional overrides

After reviewing the exact operation and backing up any work that matters:

```python
h = bash("git reset --hard", allow_destructive_git=True)
h = bash("rm -rf /reviewed/disposable/directory", allow_destructive_rm=True)
```

Both keywords are optional, keyword-only, and require actual booleans. They
bypass only their respective check for that call, including its command prefix.
They are **not** user-approval tokens. No environment variable disables these
checks, including historical `PI_BASH_ALLOW_DESTRUCTIVE_*` variables. Both
explicit flags together bypass scanning; otherwise oversized or unparseable
commands must be simplified. Never use an override merely to silence a warning.

## Bounds and platform behavior

Scanning is limited to 65,536 characters and 4,096 tokens. A call permits at most
eight Git status probes under one two-second deadline, plus bounded cleanup.
The POSIX probe uses direct argv, disables Git fsmonitor, and does not run the
command prefix. Nonblocking pipe reads retain at most 65,537 bytes; excess
output or timeout refuses the command. There is no reader thread that can hang
on a descendant holding a pipe open. Cleanup kills the probe process group and
waits at most 200 ms on failure while the leader is still unreaped. A successfully
reaped probe is never signalled by PID/PGID afterward, avoiding identity reuse.
The normal BashHandle lifecycle is unchanged.

On Windows, detected destructive Git commands refuse because this change does
not provide a bounded native Git probe or Git Bash path translation. Detected
recursive-force rm also refuses there for the same path-translation reason.
Explicit per-call overrides are available after review. Ordinary commands do
not need a probe. POSIX host tests do not establish live Windows behavior.
Commands executed *inside* an SSH payload are not checked as remote commands;
the guard does not reinterpret local paths as paths on the remote machine.

## Deliberate limits and threat model

This is a smaller native implementation inspired by upstream #2373/#2384, not
a port of their multi-thousand-line shell approximator or a parity claim.
The parser does not emulate shell execution. In particular it does not resolve
aliases/functions, computed command names, ANSI-C/locale quoting, expansions,
arrays, nested interpreters (`sh -c`, `env -S`), eval payloads, traps, shell startup files,
heredoc programs, scripts read from files, `xargs`/`find -exec`, or generated code.
These forms can evade detection. Argument-taking wrapper options cover only the
listed common `sudo`/`env` options in the scanner; unknown option grammars can
also evade detection. A single positional `git checkout name` is ambiguous
between a branch switch and a file discard and is not covered; use the explicit
`git checkout -- path` form for a checked discard. Quoted operator characters and unfamiliar
shell syntax may instead cause conservative false refusals. Commands hidden in
a remote SSH payload are likewise outside the literal direct-command scope.

The guard trusts the installed Git executable and the OS filesystem. Symlink
changes, repository edits between probe and launch, earlier commands creating
new rm targets or symlinks, slow/network filesystems, and process-global cwd or
environment changes can invalidate a pre-launch observation. The small probe
budget does not bound OS path-resolution latency. Killing a probe process group
is not proof against a malicious executable that escapes that group. No claim
of adversarial containment, race freedom, or protection against arbitrary Python
or a deliberate explicit override is made.

Tests use disposable repositories and directories, mocked launch checks, and
synthetic probe executables. They cover refusals before handle construction,
intentional overrides, dirty/staged/untracked/ignored work, literal relocation,
quoting/options, symlink escape, bounded scan/probe behavior, and normal handle
execution. No real user worktree is reset or cleaned by the tests.
