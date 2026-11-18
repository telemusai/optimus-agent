import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest

MODULE = Path(__file__).resolve().parents[1] / 'scripts/install-opencode-bun-relay.py'
spec = importlib.util.spec_from_file_location('opencode_relay_install', MODULE)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class RelayInstallTest(unittest.TestCase):
    @unittest.skipIf(sys.platform == 'win32', 'Linux installer uses flock')
    def test_preserves_other_models_auth_and_provider_settings_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as folder:
            profile = Path(folder)
            before = {'providers': {'other': {'apiKey': 'fixture-secret'}, 'opencode': {
                'apiKey': 'FIXTURE_KEY', 'baseUrl': 'https://opencode.ai/zen/v1',
                'models': [{'id': 'other-model', 'headers': {'x-test': 'keep'}}]}}}
            (profile / 'models.json').write_text(json.dumps(before))
            installer.configure_model(profile, 18997)
            installer.configure_model(profile, 18997)
            after = json.loads((profile / 'models.json').read_text())
            self.assertEqual(after['providers']['other'], before['providers']['other'])
            provider = after['providers']['opencode']
            self.assertEqual(provider['apiKey'], 'FIXTURE_KEY')
            self.assertEqual(provider['baseUrl'], 'https://opencode.ai/zen/v1')
            self.assertEqual(provider['models'][0], before['providers']['opencode']['models'][0])
            self.assertEqual(len(provider['models']), 2)
            self.assertEqual(provider['models'][1]['baseUrl'], 'http://127.0.0.1:18997/v1')
            self.assertEqual((profile / 'models.json').stat().st_mode & 0o777, 0o600)

    def test_unit_paths_are_quoted_and_reject_line_injection(self):
        self.assertEqual(installer.unit_quote('/a path/$b%/file'), '"/a path/$$b%%/file"')
        self.assertEqual(installer.unit_directory('/a path/$b%/file'), '/a path/$b%%/file')
        with self.assertRaises(ValueError):
            installer.unit_directory('relative')
        with self.assertRaises(ValueError):
            installer.unit_quote('/tmp/a\nExecStart=evil')


if __name__ == '__main__':
    unittest.main()
