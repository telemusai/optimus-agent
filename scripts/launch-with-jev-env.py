"""Load an explicit, private Jev environment file without executing shell code."""

import os
from pathlib import Path
import stat
import sys


KEY_NAMES = ("TYPESAFE_API_KEY", "JEV_API_KEY")
MAX_FILE_BYTES = 8192


def load_jev_env(path: Path, environment: dict[str, str]) -> dict[str, str]:
    if any(environment.get(name, "").strip() for name in KEY_NAMES):
        return environment.copy()
    if os.name != "posix":
        raise ValueError("Private env files require POSIX ownership checks")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return environment.copy()
    with os.fdopen(descriptor, "rb") as stream:
        metadata = os.fstat(stream.fileno())
        if (not stat.S_ISREG(metadata.st_mode)
                or metadata.st_uid != os.getuid()
                or stat.S_IMODE(metadata.st_mode) & 0o077):
            raise ValueError("Jev env file must be owned by this user with mode 600")
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("Jev env file is too large")
    values = {}
    for line in raw.decode("utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, separator, value = line.partition("=")
        name, value = name.strip(), value.strip()
        if not separator or name not in KEY_NAMES or name in values:
            raise ValueError("Jev env file requires unique TYPESAFE_API_KEY or JEV_API_KEY assignments")
        if len(value) >= 2 and value[0] in "\"'" and value[-1] == value[0]:
            value = value[1:-1]
        if not value or len(value) > 4096 or any(ord(char) < 33 or ord(char) > 126 for char in value):
            raise ValueError("Jev env file contains an invalid key value")
        values[name] = value
    return {**environment, **values}


def main() -> None:
    environment = dict(os.environ)
    path = Path(environment["PRIME_AGENT_CODING_AGENT_DIR"]) / "jev" / "env"
    try:
        environment = load_jev_env(path, environment)
    except (OSError, UnicodeError, ValueError):
        # Neither exception text nor file contents may reach the terminal.
        if os.name == "posix":
            print("Jev env file could not be loaded; check its format, owner and mode 600 permissions.", file=sys.stderr)
        else:
            print("Jev env files require POSIX permissions; use /jev key or an environment variable on Windows.", file=sys.stderr)
    os.execve(sys.argv[1], sys.argv[1:], environment)


if __name__ == "__main__":
    main()
