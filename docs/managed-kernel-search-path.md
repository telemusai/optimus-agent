# Managed search tools in the native kernel

The local `ipython` kernel receives the same managed-tool PATH default as the
Bash tool. This lets runtime `bash()` commands find provisioned `rg` and `fd`
without adding the managed directory to the user's login shell. The host PATH
is not modified. No download or kernel-environment rebuild is triggered by this
PATH setup.

The shared tool manager still owns provisioning and directory selection:
`PI_BIN_DIR`, then `PI_CODING_AGENT_DIR` or `PRIME_AGENT_CODING_AGENT_DIR` plus
`bin`, then the existing home-directory default. The managed directory comes
first when absent; an existing entry is not duplicated or reordered.
Explicit kernel `env` PATH overrides, including an empty PATH, remain authoritative.
PATH keys are case-sensitive on POSIX and case-insensitive on Windows. The shared
shell helper retains its existing empty-component filtering when prepending the
managed directory. Shell selection remains
independent of PATH and retains the trusted absolute-shell defaults.

This applies to the native local kernel. An Optimus process launched on an SSH
host uses that host's environment and managed directory. Running `ssh host ...`
from a local kernel does not provision remote tools or inject the local PATH
into the remote shell. There is no native remote-kernel adapter changed here.

The prompt keeps the Jev candidate workflow (`rg --json`, `from_ripgrep`,
`present`) and adds `fd` for file discovery. Search helpers remain optional.

Offline regressions exercise platform PATH rules, explicit overrides, and a
recording kernel factory followed by a real shell subprocess resolving fake
`rg` and `fd` from an isolated directory with a clean PATH. These tests do not
claim remote SSH or a full Python-kernel startup validation.
