import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "launch-with-jev-env.py"
SPEC = importlib.util.spec_from_file_location("jev_launcher", SCRIPT)
LAUNCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(LAUNCHER)


@unittest.skipUnless(os.name == "posix", "Linux launcher")
class JevEnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "jev" / "env"
        self.path.parent.mkdir()

    def write(self, text):
        self.path.write_text(text)
        self.path.chmod(0o600)

    def test_loads_assignments_and_preserves_other_environment(self):
        self.write('# key\nTYPESAFE_API_KEY="synthetic-primary"\nJEV_API_KEY=synthetic-alias\n')
        result = LAUNCHER.load_jev_env(self.path, {"UNCHANGED": "yes"})
        self.assertEqual(result, {"UNCHANGED": "yes", "TYPESAFE_API_KEY": "synthetic-primary", "JEV_API_KEY": "synthetic-alias"})

    def test_existing_environment_credential_wins(self):
        self.write("TYPESAFE_API_KEY=synthetic-file\n")
        existing = {"JEV_API_KEY": "synthetic-process"}
        self.assertEqual(LAUNCHER.load_jev_env(self.path, existing), existing)

    def test_refuses_unsafe_permissions_symlinks_and_oversized_files(self):
        self.write("JEV_API_KEY=synthetic\n")
        self.path.chmod(0o644)
        with self.assertRaises(ValueError):
            LAUNCHER.load_jev_env(self.path, {})
        target = self.path.with_name("target")
        self.path.rename(target)
        self.path.symlink_to(target)
        with self.assertRaises(OSError):
            LAUNCHER.load_jev_env(self.path, {})
        self.path.unlink()
        self.write("x" * 8193)
        with self.assertRaises(ValueError):
            LAUNCHER.load_jev_env(self.path, {})

    def test_does_not_execute_shell_content(self):
        marker = Path(self.directory.name) / "marker"
        self.write(f"JEV_API_KEY=$(touch {marker})\n")
        with self.assertRaises(ValueError):
            LAUNCHER.load_jev_env(self.path, {})
        self.assertFalse(marker.exists())

    def test_exec_preserves_arguments_cwd_and_does_not_print_key(self):
        self.write("JEV_API_KEY=synthetic-secret\n")
        environment = {name: value for name, value in os.environ.items() if name not in LAUNCHER.KEY_NAMES}
        environment["PRIME_AGENT_CODING_AGENT_DIR"] = self.directory.name
        result = subprocess.run([sys.executable, str(SCRIPT), sys.executable, "-c",
            "import os,sys; assert os.environ['JEV_API_KEY']=='synthetic-secret'; assert sys.argv[1]=='with spaces'; print('loaded')",
            "with spaces"], env=environment, cwd=self.directory.name, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, "loaded\n")
        self.assertEqual(result.stderr, "")

    def test_invalid_file_warns_without_echoing_contents_and_still_launches(self):
        self.write("unexpected=synthetic-secret\n")
        environment = {name: value for name, value in os.environ.items() if name not in LAUNCHER.KEY_NAMES}
        environment["PRIME_AGENT_CODING_AGENT_DIR"] = self.directory.name
        result = subprocess.run([sys.executable, str(SCRIPT), sys.executable, "-c", "print('started')"],
            env=environment, capture_output=True, text=True, check=True)
        self.assertEqual(result.stdout, "started\n")
        self.assertIn("could not be loaded", result.stderr)
        self.assertNotIn("synthetic-secret", result.stderr)


if __name__ == "__main__":
    unittest.main()
