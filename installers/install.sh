#!/bin/sh
# Install/update the newest stable version tag. Builds Rust locally.
set -eu

optimus_install() {
    case "$(uname -s)" in
        Linux|Darwin) ;;
        *) printf '%s\n' 'Use install.ps1 on Windows.' >&2; return 1 ;;
    esac
    for optimus_command in curl git cargo uv python3; do
        command -v "$optimus_command" >/dev/null 2>&1 || {
            printf 'Missing %s. See https://github.com/telemusai/optimus-agent#install-and-update for prerequisites.\n' "$optimus_command" >&2
            return 1
        }
    done
    python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else "Python 3.11+ is required")'
    optimus_temporary=$(mktemp -d)
    trap 'rm -rf "$optimus_temporary"' EXIT HUP INT TERM
    curl --proto '=https' --proto-redir '=https' --tlsv1.2 -fsSL https://telemus.ai/optimus-agent/install.py -o "$optimus_temporary/install.py"
    python3 "$optimus_temporary/install.py" "$@"
}

optimus_install "$@"
