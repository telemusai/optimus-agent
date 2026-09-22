# Telegram

Run `/telegram` in Prime's terminal and select **Setup with BotFather**. The bot connects to the session in which you run setup. It uses the same model, tools, goals, and compaction as that session, including provider-side compaction where supported.

1. Open the official [@BotFather](https://t.me/BotFather) in Telegram.
2. Send `/newbot`, choose a display name, and choose a username ending in `bot`.
3. Copy the token into Prime's setup input dialog. Do not paste it into a regular chat prompt or pass it as a slash-command argument. The input is visible on screen, but is not added to the model transcript.
4. Open Prime's one-time pairing link and press **Start** in your new bot's private chat. The link expires in ten minutes; `/telegram pair` creates a fresh link.
5. Send the bot a text message. Use `/help` in Telegram to see commands.

The paired account controls the connected Prime session and its tools with your local account's permissions. Pair only your own account and keep the bot token and pairing link private. This version supports one bot and one paired private account per Prime profile; groups, topics, photos, documents, and voice messages are not supported.

## Connection controls in Prime

| Command | Effect |
| --- | --- |
| `/telegram` | Open the connection menu |
| `/telegram setup` | Show BotFather instructions and configure a bot |
| `/telegram status` | Show bot, pairing, worker status, connected session, and last error |
| `/telegram pair` | Refresh an unused pairing link |
| `/telegram here` | Connect Telegram to the current terminal session |
| `/telegram restart` | Enable and restart the connection |
| `/telegram pause` | Stop polling and disable automatic startup, retaining configuration |
| `/telegram disconnect` | Stop polling and remove local Telegram credentials, pairing, and delivery state |

Once paired, replace an account by disconnecting and setting up again. A second account cannot replace the paired account using `/start`.

The connector runs in a background process and stays connected when you close the terminal. The computer and Prime daemon must remain running. After a computer or daemon restart, opening Prime starts an enabled connector again. This is not an operating-system boot service. Use `/telegram restart` if the connection stopped after an error. `--no-extensions` disables this built-in command and its automatic startup.

The current connector requires the Node package installation; standalone Bun binaries report an explicit unsupported-installation error. Polling needs outbound HTTPS to Telegram and no public listener or inbound firewall port. Existing webhooks are detected and left intact; disconnect the other integration or create a separate bot. Only one poller can use a bot token.

## Commands in Telegram

Session commands use Prime's existing session APIs rather than sending administrative requests as ordinary model prompts.

| Commands | Usage |
| --- | --- |
| `/help`, `/session` | Help and current session status |
| `/new [--name "name"] [prompt]`, `/clear` | Start a session |
| `/resume [session-id\|path.jsonl]` | List saved sessions or resume one |
| `/name [name]`, `/rename` | Show or set the session name |
| `/model [provider/model-id]` | List/search available models or select an exact model |
| `/effort [level]`, `/thinking` | Show or set a supported reasoning level |
| `/fast [on\|off]` | Toggle fast mode on models that support it |
| `/context`, `/usage` | Token, cost, and context statistics |
| `/compact [instructions]`, `/refine [instructions]` | Use Prime's compaction and refinement |
| `/goal`, `/autonomous` | Use Prime's goal and autonomous-mode commands |
| `/heartbeat`, `/heartbeats` | Configure the session heartbeat or list heartbeats |
| `/rlm_max_depth [integer]` | Show or set this session's RLM depth |
| `/fork [entry-id]`, `/tree [entry-id]` | List eligible entries or fork/navigate |
| `/reload`, `/system_prompt`, `/copy` | Reload resources, show the system prompt, or show the last answer |
| `/settings` | Show current settings; use the commands above to change model or effort |
| `/stop`, `/cancel` | Interrupt work and compaction and clear pending input |

Hyphenated Prime names such as `/system-prompt` also work when typed manually; Telegram's menu uses underscores. Registered extension commands, prompt templates, and `/skill:name` can also run. Login, update, package management, terminal display commands, and unsupported built-ins direct you back to the terminal.

New text arriving during a turn is queued as a follow-up. Replies contain assistant text and visible session notices; raw tool output and thinking blocks are not mirrored. Long replies are split into plain-text messages. To review the full transcript, use Prime's terminal.

Extension confirmations, selections, and text questions appear with a short request ID. Reply with `/answer <id> yes`, `no`, an option number, or text as instructed. `/answer <id> cancel` dismisses the question. Questions expire after five minutes and are never automatically approved. Full terminal editor requests are cancelled with an explanation.

## Storage and recovery

Telegram data lives under `~/.prime/agent/telegram/` (or `telegram/` within `PRIME_AGENT_CODING_AGENT_DIR`). Setup does not rewrite `models.json`, `settings.json`, or `auth.json`. Model and session commands subsequently have their normal Prime effects.

`connection.json` contains the bot token, paired account, and session binding. `state.json` holds the poll offset and pending text deliveries; worker and stop files coordinate the background process. Files are created with mode `0600` and the directory with `0700` on POSIX. On Windows, use a private user profile with appropriate filesystem ACLs.

Processed updates are deduplicated across restarts. If Prime crashes during prompt admission, the uncertain prompt is not replayed automatically; a notice recommends checking `/session` and `/copy` before repeating it. Pending replies are retried with Telegram's rate-limit delay. A network failure after Telegram accepted a reply but before Prime recorded it can result in a duplicate reply; the Bot API does not provide idempotent sends. Output produced while the connector is stopped can be recovered with `/copy` or the terminal transcript.

If a token is compromised, revoke it in BotFather and run setup with a new token. Disconnecting removes Prime's local copy; it does not revoke the token at Telegram.

The design follows vr-ai-chat's private channel routing, durable poll offsets, and Bot API delivery patterns. Telegram references: [bot creation](https://core.telegram.org/bots/tutorial), [Bot API](https://core.telegram.org/bots/api), and [deep links](https://core.telegram.org/bots/features#deep-linking).
