from __future__ import annotations

import importlib
import os
import shlex
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from rlm.bash_guard import (guard_command, DestructiveGitRefusalError,
                            DestructiveRmRefusalError, BashGuardRefusalError)

guard = importlib.import_module("rlm.bash_guard")
bash_module = importlib.import_module("rlm.bash")


class GuardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.env = {"PATH": os.defpath, "HOME": str(self.root)}

    def check(self, command, **kwargs):
        return guard_command(command, cwd=str(self.workspace), env=self.env, **kwargs)

    def test_non_destructive_and_quoted_data_do_not_probe(self):
        commands = ["git status", "git log --oneline", "git reset --soft HEAD", "git checkout -b topic",
                    "git clean -fn", "git clean -nf -- ignored", "git clean --dry-run -f", "git clean -- -f",
                    "rm -r folder", "rm -f file", "rm -r folder; rm -f file", "rm -- -rf",
                    "echo 'git reset --hard'", "printf '%s' 'rm -rf /'", "# git reset --hard\necho ok"]
        with mock.patch.object(guard, "_probe") as probe:
            for command in commands:
                with self.subTest(command=command):
                    self.check(command)
            probe.assert_not_called()

    def test_git_taxonomy_detects_direct_literal_discard_forms(self):
        commands = ["git reset --hard", "git reset -q HEAD --hard", "git restore .", "git restore --staged tracked",
                    "git checkout -- .", "git checkout HEAD -- tracked", "git checkout HEAD tracked",
                    "git checkout -f", "git checkout --ours tracked", "git checkout --conflict=merge tracked",
                    "git clean -fd", "git clean --force", "git clean folder -f", "'git' reset --hard",
                    "g\\it reset --hard", "/usr/bin/git reset --hard", "git re" + chr(92) + "\nset --hard",
                    "echo ok # comment\ngit reset --hard", "echo ok;\ngit reset --hard"]
        with mock.patch.object(guard, "_probe", return_value=[" M tracked"]):
            for command in commands:
                with self.subTest(command=command), self.assertRaises(DestructiveGitRefusalError):
                    self.check(command)

    def test_rm_options_quotes_and_protected_paths(self):
        outside = self.root / "outside"
        outside.mkdir()
        (self.workspace / "link").symlink_to(outside, target_is_directory=True)
        commands = ["rm -rf /", "rm --recursive --force ..", "rm sub -Rf ../outside", "rm -r -f .git",
                    "rm -fr .env", "rm -rf .", "rm -rf link", "rm -rf '$HOME'", "rm -rf $PWD/sub",
                    "rm -rf *", "rm -rf -", "rm -rf", "rm -rf -- ../outside", "rm -rf --unknown sub",
                    "'rm' -rf /", "r\\m -rf /", "rm -r sub -f /", "echo ok # comment\nrm -rf /",
                    'rm -rf "' + str(outside) + '"']
        for command in commands:
            with self.subTest(command=command), self.assertRaises(DestructiveRmRefusalError):
                self.check(command)
        for command in ["rm -rf sub", "rm -fr ./sub", "rm sub -r -f", "rm -rf 'my dir'", "rm -rf -- -name",
                        "rm --recursive --force sub", "rm -rf sub/nested", "rm -rf missing"]:
            with self.subTest(command=command):
                self.check(command)

    def test_home_is_protected_even_when_nested_under_the_working_directory(self):
        self.env["HOME"] = str(self.workspace / "home")
        with self.assertRaises(DestructiveRmRefusalError):
            self.check("rm -rf home")
        with self.assertRaises(DestructiveRmRefusalError):
            guard_command("rm -rf /tmp", cwd="/", env=self.env)

    def test_relocations_and_uncertain_syntax(self):
        for command, suffix in [("cd nested && git reset --hard", "nested"),
                                ("git -C nested -C deep reset --hard", "nested/deep")]:
            with mock.patch.object(guard, "_probe", return_value=[]) as probe:
                self.check(command)
                self.assertEqual(probe.call_args.args[0], self.workspace / suffix)
        for command in ["cd nested; git reset --hard", "cd nested || git reset --hard", "pushd nested && git reset --hard",
                        "git --git-dir elsewhere reset --hard", "git -c core.worktree=elsewhere reset --hard",
                        "GIT_DIR=elsewhere git reset --hard", "(git reset --hard)", "git reset --hard > log"]:
            with self.subTest(command=command), self.assertRaises(DestructiveGitRefusalError), mock.patch.object(guard, "_probe") as probe:
                self.check(command)
            probe.assert_not_called()
        self.check("cd nested && rm -rf sub")
        for command in ["cd nested; rm -rf sub", "rm -rf sub | cat", "env rm -rf sub"]:
            with self.subTest(command=command), self.assertRaises(DestructiveRmRefusalError):
                self.check(command)

    def test_cdpath_relative_cd_refuses_without_probing_an_assumed_repository(self):
        self.env["CDPATH"] = str(self.root / "another-parent")
        for command in ["cd nested && git reset --hard", "cd ./nested && git restore ."]:
            with self.subTest(command=command), mock.patch.object(guard, "_probe") as probe:
                with self.assertRaises(DestructiveGitRefusalError):
                    self.check(command)
                probe.assert_not_called()
        with self.assertRaises(DestructiveRmRefusalError):
            self.check("cd nested && rm -rf sub")
        with mock.patch.object(guard, "_probe", return_value=[]) as probe:
            self.check("cd " + shlex.quote(str(self.workspace)) + " && git reset --hard")
            self.assertEqual(probe.call_args.args[0], self.workspace)

    def test_assignment_only_prefix_makes_later_targets_uncertain(self):
        for command in ["CDPATH=/other; cd nested && git reset --hard",
                        "GIT_DIR=/other; git reset --hard", "HOME=/other\ngit clean -f"]:
            with self.subTest(command=command), mock.patch.object(guard, "_probe") as probe:
                with self.assertRaises(DestructiveGitRefusalError):
                    self.check(command)
                probe.assert_not_called()
        with self.assertRaises(DestructiveRmRefusalError):
            self.check("CDPATH=/other; cd nested && rm -rf sub")

    def test_preceding_commands_and_argument_taking_wrappers_refuse(self):
        for command in ["touch tracked; git reset --hard", "git status && git reset --hard",
                        "echo checked\ngit restore .", "sudo -u root git reset --hard",
                        "env -C /tmp git reset --hard"]:
            with self.subTest(command=command), mock.patch.object(guard, "_probe", return_value=[]) as probe:
                with self.assertRaises(DestructiveGitRefusalError):
                    self.check(command)
                probe.assert_not_called()
        for command in ["sudo -u root rm -rf /", "sudo --group staff rm -rf /", "env -C /tmp rm -rf /", "env -u HOME rm -rf /"]:
            with self.subTest(command=command), self.assertRaises(DestructiveRmRefusalError):
                self.check(command)

    def test_overrides_are_exact_boolean_per_call_only(self):
        for bad in [1, "true", None, []]:
            for key in ["allow_destructive_git", "allow_destructive_rm"]:
                with self.assertRaises(TypeError):
                    self.check("echo ok", **{key: bad})
        with mock.patch.dict(os.environ, {"PI_BASH_ALLOW_DESTRUCTIVE_RM": "1", "PI_BASH_ALLOW_DESTRUCTIVE_GIT": "1"}):
            with self.assertRaises(DestructiveRmRefusalError):
                self.check("rm -rf /")
            with mock.patch.object(guard, "_probe", return_value=[" M tracked"]), self.assertRaises(DestructiveGitRefusalError):
                self.check("git reset --hard")
        self.check("rm -rf /", allow_destructive_rm=True)
        self.check("git reset --hard", allow_destructive_git=True)
        with self.assertRaises(DestructiveRmRefusalError):
            self.check("rm -rf /", allow_destructive_git=True)

    def test_refusal_precedes_handle_and_prefix_is_checked(self):
        with mock.patch.object(bash_module, "BashHandle") as handle, mock.patch.object(bash_module.os, "getcwd", return_value=str(self.workspace)):
            with self.assertRaises(DestructiveRmRefusalError):
                bash_module.bash("rm -rf /")
            handle.assert_not_called()
            with mock.patch.dict(os.environ, {"PRIME_AGENT_BASH_COMMAND_PREFIX": "rm -rf /"}):
                with self.assertRaises(DestructiveRmRefusalError):
                    bash_module.bash("echo safe")
            handle.assert_not_called()
            with mock.patch.dict(os.environ, {"PRIME_AGENT_BASH_COMMAND_PREFIX": ""}):
                bash_module.bash("echo unchanged")
                handle.assert_called_once_with("echo unchanged")
                handle.reset_mock()
                bash_module.bash("rm -rf /", allow_destructive_rm=True)
                handle.assert_called_once_with("rm -rf /")

    def test_scan_and_probe_count_bounds(self):
        with self.assertRaises(BashGuardRefusalError):
            self.check("echo " + "x" * (guard.MAX_COMMAND + 1))
        with self.assertRaises(BashGuardRefusalError):
            self.check("echo " + "x " * (guard.MAX_TOKENS + 1))
        with mock.patch.object(guard, "_probe", return_value=[]), self.assertRaises(DestructiveGitRefusalError):
            self.check(";".join(["git reset --hard"] * (guard.MAX_PROBES + 1)))

    def test_windows_probe_fails_safely_only_on_destructive_git(self):
        with mock.patch.object(guard.os, "name", "nt"), self.assertRaises(DestructiveGitRefusalError):
            guard._probe(self.workspace, False, self.env, time.monotonic() + 1)
        with mock.patch.object(guard, "_probe") as probe:
            self.check("git status")
            probe.assert_not_called()

        with mock.patch.object(guard, "Path", return_value=self.workspace), mock.patch.object(guard.os, "name", "nt"):
            with self.assertRaises(DestructiveRmRefusalError):
                self.check("rm -rf sub")
            self.check("rm -r sub")

    def test_documented_indirect_commands_are_not_claimed_as_covered(self):
        with mock.patch.object(guard, "_probe") as probe:
            for command in ["sh -c 'git reset --hard'", "ssh host 'rm -rf /remote'", "G=git; $G reset --hard"]:
                self.check(command)
            probe.assert_not_called()

    def test_probe_failure_and_routing_env_refuse(self):
        for key in ["GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_CONFIG_COUNT"]:
            with self.subTest(key=key), self.assertRaises(DestructiveGitRefusalError):
                guard._probe(self.workspace, False, {**self.env, key: "x"}, time.monotonic() + 1)
        with mock.patch.object(guard.shutil, "which", return_value=None), self.assertRaises(DestructiveGitRefusalError):
            guard._probe(self.workspace, False, self.env, time.monotonic() + 1)
        with mock.patch.object(guard.subprocess, "Popen", side_effect=OSError), self.assertRaises(DestructiveGitRefusalError):
            guard._probe(self.workspace, False, self.env, time.monotonic() + 1)

    def test_refusal_elides_long_paths_and_lists(self):
        with mock.patch.object(guard, "_probe", return_value=[" M " + "x" * 1000] * 30):
            with self.assertRaises(DestructiveGitRefusalError) as error:
                self.check("git reset --hard")
        self.assertLess(len(str(error.exception)), 2200)
        self.assertIn("more paths omitted", str(error.exception))
        self.assertIn("staging alone is not enough", str(error.exception))


@unittest.skipUnless(os.name == "posix" and shutil.which("git"), "POSIX git fixture")
class GitFixtureTests(GuardTests):
    def setUp(self):
        super().setUp()
        self.git = shutil.which("git")
        self.run_git("init", "-q")
        (self.workspace / "tracked").write_text("original")
        self.run_git("add", "tracked")
        self.run_git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-qm", "initial")

    def run_git(self, *args):
        return subprocess.run([self.git, *args], cwd=self.workspace, env=self.env,
                              stdin=subprocess.DEVNULL, capture_output=True, check=True, timeout=3)

    def test_successful_probe_does_not_signal_a_reaped_process_group(self):
        with mock.patch.object(guard.os, "killpg") as kill:
            self.check("git reset --hard")
            kill.assert_not_called()

    def test_dirty_and_clean_repository_and_ignored_files(self):
        self.check("git reset --hard")
        (self.workspace / "tracked").write_text("keep my work")
        for command in ["git reset --hard", "git checkout -- tracked", "git restore tracked", "git clean -fd"]:
            with self.subTest(command=command), self.assertRaises(DestructiveGitRefusalError):
                self.check(command)
            self.assertEqual((self.workspace / "tracked").read_text(), "keep my work")
        self.run_git("add", "tracked")
        with self.assertRaises(DestructiveGitRefusalError):
            self.check("git reset --hard")
        self.run_git("-c", "user.name=Fixture", "-c", "user.email=fixture@invalid", "commit", "-qm", "update")
        (self.workspace / ".git" / "info" / "exclude").write_text("ignored\n")
        (self.workspace / "ignored").write_text("keep")
        self.check("git clean -f")
        with self.assertRaises(DestructiveGitRefusalError):
            self.check("git clean -fx")
        self.run_git("config", "status.showUntrackedFiles", "no")
        (self.workspace / "untracked").write_text("keep")
        with self.assertRaises(DestructiveGitRefusalError):
            self.check("git clean -f")

    def test_real_bash_safe_cleanup_and_explicit_git_override_in_fixture(self):
        import asyncio
        from rlm import bash
        (self.workspace / "scratch").mkdir()
        (self.workspace / "scratch" / "file").write_text("temporary")
        (self.workspace / "tracked").write_text("intentional disposable change")
        with mock.patch.dict(os.environ, self.env, clear=True), mock.patch.object(bash_module.os, "getcwd", return_value=str(self.workspace)):
            with self.assertRaises(DestructiveGitRefusalError):
                bash("git reset --hard")
            async def run():
                cleaned = await bash("rm -rf scratch")
                reset = await bash("git reset --hard", allow_destructive_git=True)
                return cleaned, reset
            cleaned, reset = asyncio.run(run())
        self.assertEqual(cleaned.exit_code, 0)
        self.assertEqual(reset.exit_code, 0)
        self.assertFalse((self.workspace / "scratch").exists())
        self.assertEqual((self.workspace / "tracked").read_text(), "original")

    def test_probe_output_and_time_are_bounded(self):
        import sys
        for body, expected in [("import os; os.write(1, b'x' * 100000)", "output limit"),
                               ("import time; time.sleep(10)", "time limit")]:
            fake = self.root / "git-fixture"
            fake.write_text("#!" + sys.executable + "\n" + body + "\n")
            fake.chmod(0o755)
            start = time.monotonic()
            with mock.patch.object(guard.shutil, "which", return_value=str(fake)):
                with self.assertRaisesRegex(DestructiveGitRefusalError, expected):
                    guard._probe(self.workspace, False, self.env, start + .15)
            self.assertLess(time.monotonic() - start, 1)


if __name__ == "__main__":
    unittest.main()
