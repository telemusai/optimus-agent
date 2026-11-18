#!/usr/bin/env python3
"""Install the optional Bun transport and one free model for the current user."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request

MODEL = 'deepseek-v4.1-flash-free'
SERVICE = 'optimus-opencode-bun-relay.service'


def configure_model(profile, port):
    import fcntl
    profile.mkdir(mode=0o700, parents=True, exist_ok=True)
    path = profile / 'models.json'
    with (profile / 'models.json.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(path.read_text()) if path.exists() else {}
        provider = data.setdefault('providers', {}).setdefault('opencode', {})
        # Auth remains in the user's existing provider/auth configuration.
        if not provider:
            provider.update(api='openai-completions', baseUrl='https://opencode.ai/zen/v1')
        models = provider.setdefault('models', [])
        model = next((m for m in models if m['id'] == MODEL), None)
        if model is None:
            model = {'id': MODEL, 'name': 'DeepSeek V4.1 Flash Free', 'reasoning': True,
                     'input': ['text', 'image'], 'contextWindow': 200000, 'maxTokens': 32000,
                     'cost': {'input': 0, 'output': 0, 'cacheRead': 0, 'cacheWrite': 0}}
            models.append(model)
        model.update(api='openai-completions', baseUrl=f'http://127.0.0.1:{port}/v1')
        model.setdefault('compat', {}).update(thinkingFormat='deepseek',
            requiresReasoningContentOnAssistantMessages=True, supportsDeveloperRole=False,
            supportsStrictMode=False, maxTokensField='max_tokens')
        fd, temp = tempfile.mkstemp(prefix='.models-opencode-', dir=profile)
        try:
            with os.fdopen(fd, 'w') as out:
                json.dump(data, out, indent=2)
                out.write('\n')
                out.flush()
                os.fsync(out.fileno())
            os.replace(temp, path)
        finally:
            if os.path.exists(temp):
                os.unlink(temp)


def unit_quote(value):
    text = str(value)
    if '\n' in text or '\r' in text:
        raise ValueError('Newlines are not supported in service paths')
    return '"' + text.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%').replace('$', '$$') + '"'


def unit_directory(path):
    text = str(path)
    if not text.startswith('/') or '\n' in text or '\r' in text:
        raise ValueError('An absolute single-line working directory is required')
    return text.replace('%', '%%')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=18997)
    parser.add_argument('--profile', type=Path, default=Path.home() / '.config/optimus-rust')
    parser.add_argument('--workspace', type=Path, default=Path.cwd())
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('This installer uses Linux user systemd; see the manual Bun instructions for other hosts.')
    if not 1 <= args.port <= 65535:
        parser.error('Port must be between 1 and 65535')
    bun = shutil.which('bun')
    if not bun:
        parser.error('Install Bun before running this installer')
    auth = Path.home() / '.local/share/opencode/auth.json'
    key = json.loads(auth.read_text()).get('opencode', {}).get('key')
    if not isinstance(key, str) or not key or key.startswith(('PASTE_', 'YOUR_')):
        parser.error('A valid key is required in the existing OpenCode auth.json')
    source = Path(__file__).resolve().with_name('opencode-bun-relay.mjs')
    destination = Path.home() / '.local/share/optimus-opencode-relay'
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copyfile(source, destination / source.name)
    units = Path.home() / '.config/systemd/user'
    units.mkdir(parents=True, exist_ok=True)
    unit = f'''[Unit]
Description=Optimus OpenCode free DeepSeek Bun transport
After=network-online.target

[Service]
Type=simple
WorkingDirectory={unit_directory(args.workspace.resolve())}
ExecStart={unit_quote(bun)} {unit_quote(destination / source.name)}
Environment=OPENCODE_RELAY_PORT={args.port}
Restart=on-failure
RestartSec=5
UMask=0077
NoNewPrivileges=true

[Install]
WantedBy=default.target
'''
    (units / SERVICE).write_text(unit)
    subprocess.run(['systemctl', '--user', 'daemon-reload'], check=True)
    subprocess.run(['systemctl', '--user', 'enable', SERVICE], check=True)
    subprocess.run(['systemctl', '--user', 'restart', SERVICE], check=True)
    for _ in range(50):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{args.port}/health', timeout=1) as response:
                healthy = json.load(response)
            if healthy.get('model') == MODEL and healthy.get('transport') == 'bun':
                break
        except (OSError, ValueError):
            pass
        time.sleep(0.2)
    else:
        raise RuntimeError('Relay did not become healthy; model configuration was left unchanged')
    configure_model(args.profile, args.port)
    print(f'Installed {SERVICE}; configured opencode/{MODEL} in {args.profile / "models.json"}')
    print('Existing sessions keep their current model snapshot. Start a new Optimus session to use the relay.')


if __name__ == '__main__':
    main()
