# OpenCode free DeepSeek through Bun

This optional sidecar routes `opencode/deepseek-v4.1-flash-free` through Bun's actual HTTPS implementation. Optimus remains the agent and executes its own native tools. The relay performs no tool execution and has no paid-model or OpenCode Go fallback.

The verified client contract uses Bun 1.4.4, OpenCode 1.19.5 headers, timestamped 26-character session/message IDs, the account organization header, and the OpenCode environment block. Native system messages, tool schemas, arguments, results and history remain intact. Missing `bash`, `read` and `edit` names become aliases of the native `ipython` schema; responses map those aliases back to `ipython` before Optimus executes them. Existing native tools with those names are preserved. Requests must use `stream: true`, as Optimus does for both conversation and completion calls. Tool-free completion/compaction requests disable tools with `tool_choice: none`; unexpected tool calls fail closed. Other tool modes without `ipython` or the three native names are explicitly rejected.

## Linux installation

Prerequisites: Bun on PATH; the existing OpenCode API key in `~/.local/share/opencode/auth.json`; the same key configured for Optimus's `opencode` provider; a user systemd session.

```sh
python3 scripts/install-opencode-bun-relay.py --workspace /path/to/your/project
optimus-agent --provider opencode --model deepseek-v4.1-flash-free
```

The installer copies the relay to `~/.local/share/optimus-opencode-relay/`, enables `optimus-opencode-bun-relay.service`, checks local health, and atomically adds/updates only this model's endpoint in `~/.config/optimus-rust/models.json`. It preserves provider authentication, other models and the selected default model. Existing sessions retain their loaded model configuration; start a new session. The relay environment identifies its service working directory; the native Optimus session instructions remain authoritative for the actual tool workspace.

```sh
systemctl --user status optimus-opencode-bun-relay.service
journalctl --user -u optimus-opencode-bun-relay.service
curl http://127.0.0.1:18997/health
```

Health confirms relay readiness, not live provider entitlement. Upstream 401/429 responses and Retry-After pass through unchanged. Provider contract changes may require a relay update.

## Manual launch and other desktop platforms

```sh
bun scripts/opencode-bun-relay.mjs
```

Optional environment variables: `OPENCODE_AUTH_FILE`, `OPENCODE_API_KEY`, `OPENCODE_RELAY_PORT` (default 18997). Configure this model's OpenAI-compatible base URL as `http://127.0.0.1:18997/v1`. The same bearer API key authenticates loopback requests. The service binds only IPv4 loopback, rejects browser origins and other hosts, never follows upstream redirects, bounds request/tool payloads, and forwards cancellation. It does not log prompts, arguments or credentials. Non-Linux hosts can run the Bun sidecar manually; the systemd installer is Linux-only and was the validated deployment.

## Validation

```sh
bun test scripts/opencode-bun-relay.test.mjs
python3 -m unittest discover -s tests -p test_opencode_bun_relay_install.py
bash scripts/check.sh
```

The path-scoped OpenCode Bun relay CI workflow runs the focused suites on Linux with Bun 1.4.4. Regression tests use synthetic credentials and local fixtures. Live acceptance must include a native tool call, its result, and a subsequent assistant response, not only an HTTP 200. The host acceptance on 2026-10-31 ran `print(19 + 23)` in Optimus's persistent Python kernel and returned `NATIVE_TOOL_OK=42` using the free model.

To disable the transport, stop/disable the user service and remove this model's loopback `baseUrl` override (or remove the custom free-model entry). No changes to the Rust daemon protocol, package runtime, 3D environment or Web dashboard are required.

Client references: [Bun fetch](https://bun.sh/docs/runtime/networking/fetch), [OpenCode identifier source](https://github.com/anomalyco/opencode/blob/v1.19.5/packages/opencode/src/id/id.ts). The free tier's availability remains controlled by OpenCode.
