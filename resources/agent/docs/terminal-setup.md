# Terminal Setup

Prime Agent uses the [Kitty keyboard protocol](https://sw.kovidgoyal.net/kitty/keyboard-protocol/) for reliable modifier key detection. Most modern terminals support this protocol, but some require configuration.

## Kitty, iTerm2

Work out of the box.

## Ghostty

Add to your Ghostty config (`~/Library/Application Support/com.mitchellh.ghostty/config` on macOS, `~/.config/ghostty/config` on Linux):

```
keybind = alt+backspace=text:\x1b\x7f
```

Older Claude Code versions may have added this Ghostty mapping:

```
keybind = shift+enter=text:\n
```

That mapping sends a raw linefeed byte. Inside Prime Agent, that is indistinguishable from `Ctrl+J`, so tmux and Prime Agent no longer see a real `shift+enter` key event.

If Claude Code 2.x or newer is the only reason you added that mapping, you can remove it, unless you want to use Claude Code in tmux, where it still requires that Ghostty mapping.

If you want `Shift+Enter` to keep working in tmux via that remap, add `ctrl+j` to your Prime Agent `newLine` keybinding in `~/.prime/agent/keybindings.json`:

```json
{
  "newLine": ["shift+enter", "ctrl+j"]
}
```

## WezTerm

Create `~/.wezterm.lua`:

```lua
local wezterm = require 'wezterm'
local config = wezterm.config_builder()
config.enable_kitty_keyboard = true
return config
```

## VS Code (Integrated Terminal)

`keybindings.json` locations:
- macOS: `~/Library/Application Support/Code/User/keybindings.json`
- Linux: `~/.config/Code/User/keybindings.json`
- Windows: `%APPDATA%\\Code\\User\\keybindings.json`

Add to `keybindings.json` to enable `Shift+Enter` for multi-line input:

```json
{
  "key": "shift+enter",
  "command": "workbench.action.terminal.sendSequence",
  "args": { "text": "\u001b[13;2u" },
  "when": "terminalFocus"
}
```

## Windows Terminal

Optimus displays tool-result images using SIXEL when Windows Terminal 1.22 or later confirms
support. This works with native PowerShell; WSL and external image converters
are not required. Kitty and iTerm2 image protocols remain supported on compatible
terminals. The approach follows [Oh My Pi's terminal image implementation](https://github.com/can1357/oh-my-pi).

Image support is detected at startup without blocking input. SIXEL also requires
a valid terminal cell-size report so images stay inside their reserved rows. Unsupported terminals,
unanswered probes, invalid images, and images exceeding the safety limits show
`Cannot display image` instead. Fullscreen images display only when their entire
rectangle fits the transcript; clipped images, selection, and overlays use a
placeholder so graphics cannot cover the header or input area. At most eight
images are displayed in a fullscreen viewport. Images retain their reserved rows
while scrolling. tmux and screen use text fallbacks.

Optimus checks the standard device-attributes reply for SIXEL support, including
Windows Terminal's reply. When available, XTerm graphics queries also supply the
terminal's raster limits and number of colour registers. Images are quantized to
that palette size so a 16-colour terminal cannot misinterpret a 256-colour image.
Windows Terminal uses its supported 256-colour palette when no palette query is
available; other terminals start with 16 colours until they report their capacity.

To disable terminal graphics for a session in PowerShell:

```powershell
$env:PI_FORCE_IMAGE_PROTOCOL = 'off'
optimus-agent
```

On Linux/macOS: `PI_FORCE_IMAGE_PROTOCOL=off optimus-agent`. Unset the variable
to restore automatic detection. Advanced users can select `kitty`, `iterm2`, or
`sixel` for a terminal they know supports it; automatic detection is recommended.
VS Code's terminal also requires `terminal.integrated.enableImages` enabled.

Add to `settings.json` (Ctrl+Shift+, or Settings → Open JSON file) to forward the modified Enter keys Prime Agent uses:

```json
{
  "actions": [
    {
      "command": { "action": "sendInput", "input": "\u001b[13;2u" },
      "keys": "shift+enter"
    },
    {
      "command": { "action": "sendInput", "input": "\u001b[13;3u" },
      "keys": "alt+enter"
    }
  ]
}
```

- `Shift+Enter` inserts a new line.
- Windows Terminal binds `Alt+Enter` to fullscreen by default. That prevents Prime Agent from receiving `Alt+Enter` for follow-up queueing.
- Remapping `Alt+Enter` to `sendInput` forwards the real key chord to Prime Agent instead.

If you already have an `actions` array, add the objects to it. If the old fullscreen behavior persists, fully close and reopen Windows Terminal.

## XTerm image previews

For full-colour SIXEL images, enable 256 colour registers and size reports for the
XTerm window:

```bash
xterm -xrm 'XTerm*decGraphicsID: 340' \
  -xrm 'XTerm*numColorRegisters: 256' \
  -xrm 'XTerm*allowWindowOps: true' -e optimus-agent
```

Optimus also respects XTerm's default 16-colour palette, with reduced shading.

## xfce4-terminal, terminator

These terminals have limited escape sequence support. Modified Enter keys like `Ctrl+Enter` and `Shift+Enter` cannot be distinguished from plain `Enter`, preventing custom keybindings such as `submit: ["ctrl+enter"]` from working.

For the best experience, use a terminal that supports the Kitty keyboard protocol:
- [Kitty](https://sw.kovidgoyal.net/kitty/)
- [Ghostty](https://ghostty.org/)
- [WezTerm](https://wezfurlong.org/wezterm/)
- [iTerm2](https://iterm2.com/)
- [Alacritty](https://github.com/alacritty/alacritty) (requires compilation with Kitty protocol support)

## IntelliJ IDEA (Integrated Terminal)

The built-in terminal has limited escape sequence support. Shift+Enter cannot be distinguished from Enter in IntelliJ's terminal.

If you want the hardware cursor visible, set `PI_HARDWARE_CURSOR=1` before running `prime-agent` (disabled by default for compatibility).

Consider using a dedicated terminal emulator for the best experience.

### macOS Control+Option+Arrow shortcuts

Pending-message reordering defaults to `Control+Option+Up` and `Control+Option+Down`. Prime Agent accepts modern modified-arrow sequences and legacy Option-as-Meta wrapped Control+Arrow sequences. macOS VoiceOver uses Control+Option as its modifier, and system or terminal shortcuts can intercept these chords before they reach Prime Agent. If that happens, remap `app.message.moveEarlier` and `app.message.moveLater` in `~/.prime/agent/keybindings.json`.
