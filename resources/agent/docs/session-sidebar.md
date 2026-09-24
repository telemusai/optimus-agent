# Session sidebar

Chats stay in their project or folder group, including the current and recently used chats.
Generic folder names such as `source` use a useful ancestor name. Duplicate project names include a location suffix.
`*` marks each active chat. The current chat name is red. Keyboard selection does not move that current-chat indicator.

## Navigate and inspect

- At the end of the chat editor, Down focuses the sub-agent summary. Enter selects a running direct child of the current chat in Sessions. Enter again opens that child in the same workspace. If no running child remains, focus returns to the editor.
- Left at the start of the editor focuses Sessions. Right or Escape returns to chat.
- Select a project or chat, then use `Ctrl+Shift+L` to view its full project path and saved chat file. Enter copies the project path, or the saved chat file if the project path is unknown. Escape closes the view.
- You can also click the location line or the `Location / copy` footer. This works when the terminal cannot distinguish `Ctrl+Shift+L`.
- Delete/stop always requires confirmation. An active chat is stopped with history kept. A saved inactive chat uses its saved file identity. If it becomes active, deletion is refused. Escape cancels without a mutation.

## Hide or move

- `Ctrl+H` hides or restores Sessions and releases or restores its width.
- `Ctrl+M` moves Sessions to the other edge. Repeat presses alternate left and right.
- These actions keep chat state, drafts, selection, groups, and running work.

The default toggles work only when the terminal identifies the control key separately: CSI-u, modifyOtherKeys, or distinct Windows native virtual-key events. Raw Ctrl+H and Ctrl+M are indistinguishable from Backspace and Enter in legacy VT/ConPTY input. They cannot safely toggle the sidebar there. Raw Enter, Backspace, DEL, Tab, and newline remain editing/navigation input. No alternative shortcut is silently assigned.

Configure the existing profile `keybindings.json` action ids to use distinct keys if needed:

```json
{
  "app.sidebar.toggleVisibility": "ctrl+h",
  "app.sidebar.toggleSide": "ctrl+m",
  "app.sidebar.location": "ctrl+shift+l"
}
```

Safe raw custom control bindings remain supported. A remap to an ambiguous editing byte does not override Enter or Backspace.
