# Native terminal clipboard delivery

Mouse selections in the native chat view use the same clipboard backend as `/copy`
and the sign-in-link copy action. Selection geometry, sidebar metadata, and
configurable keybindings are unchanged.

- Local Linux tries `wl-copy` when `WAYLAND_DISPLAY` is set, then `xclip` and
  `xsel` when `DISPLAY` is set. Tools retain selection ownership after the launcher
  exits. Missing tools, failed writes, nonzero exits, and timeouts do not count as
  success. Each command has a two-second deadline covering stdin and exit. Copy
  runs asynchronously so a stalled backend does not block input/rendering.
- Windows `clip.exe` receives BOM-prefixed UTF-16LE so Unicode does not depend on
  the active code page. Other local backends retain UTF-8 input.
- SSH/Mosh sessions skip host clipboard tools, including forwarded X11 displays.
  They request the client terminal clipboard with OSC 52 instead. Local sessions
  also try OSC 52 after backend failure. Redirected stdout is not a terminal path.
- OSC 52 payloads are bounded to 100,000 encoded bytes. In tmux they use the DCS
  passthrough envelope with escaped inner ESC bytes. The outer terminal must allow
  OSC 52; tmux must permit passthrough (`allow-passthrough on`, where supported).
  Nested multiplexers, terminal security settings, and smaller terminal limits
  can still reject requests. The application does not change these settings.
- “Copied to local clipboard” means the local backend accepted all input and exited
  successfully. It is not a read-back assertion. “Clipboard request sent to terminal
  (unconfirmed...)” means the OSC 52 write and flush succeeded, not that the client
  clipboard changed. OSC 52 has no portable write acknowledgement. If forwarding
  is disabled, use terminal-native selection/copy or enable it in the client.

Validation uses synthetic commands, temporary output files, and in-memory writers.
It does not read or overwrite the operator clipboard. Real compositor/SSH/tmux
acceptance remains an interactive client-side check.
