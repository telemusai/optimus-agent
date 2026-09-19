# Optimus Agent

**A self-improving, multi-model harness for coding, research, and long-running work, with a native Rust application layer and a persistent Python runtime.**

Maintained by [Telemus AI](https://github.com/telemusai).

[Get started](#get-started) · [Memory and learning](#memory-that-you-can-inspect-and-correct) · [Rust migration](#the-rust-migration) · [Documentation](#documentation)

Optimus is the layer between an AI model and real work: the tools it can use, the context it retains, the agents it coordinates, and the state it recovers when a session ends or a connection drops.

It combines a persistent Python workspace, recursive subagents, inspectable memory, provider-aware context management, and background execution. Use one model to plan and review, others to investigate or build, and keep the work connected across sessions.

Optimus focuses on **native Windows reliability, stronger session continuity, Astra-aware model handling, project-scoped learning, measurable efficiency, and a Rust implementation of the application layer.** See [Foundations and acknowledgements](#foundations-and-acknowledgements) for the projects it builds on.

## Project status

| Area | Current position |
| --- | --- |
| Main implementation | Native Rust application with a Python execution runtime; TypeScript/Node.js retained as the reference |
| Rust implementation | Available on `main`, installed as `optimus-agent`; remaining parity gaps are documented |
| Platforms | macOS, Linux, and native Windows; Windows currently requires a Bash shell such as Git Bash |
| Memory | Session, project, and global harness memory, with optional selected sharing; authoritative project storage is JSON |
| Jev integration | Optional Compare (shadow) and Active modes against TypeSafe System One; Compare records recommendations, Active applies a narrow reversible subset to the next request |
| TencentDB-backed memory | An intended integration direction, not an implemented backend in the current `main` branch |

The feature descriptions below refer to `main` unless explicitly marked as development work. Some hardened Windows installations also use deployment-specific launchers and compatibility layers that are not included in a plain source checkout.

## What makes Optimus different

### A programmable workspace, not just a chat loop

The persistent Python REPL gives the agent a working environment that survives individual tool calls. It can inspect a repository, transform data, retain intermediate results, execute commands, and invoke skills through code.

Recursive language model (RLM) workflows let the agent delegate work into separate contexts rather than put every investigation into one growing conversation. Parent and child agents can exchange attributed messages and return results for review.

- Delegate independent investigations or implementation tasks in parallel.
- Use different configured models for different roles.
- Keep Python state and session history available for subsequent work.
- Extend the harness with Python skills, prompt templates, extensions, and MCP tools.

**Self-improving means improving the harness around the model—not training new model weights.** Refinement can preserve useful instructions, memories, skill descriptions, and subagent specifications for later work.

### Context management that understands the provider

Large context windows are useful, but they are not a reason to let every conversation grow indefinitely.

Optimus retains an inclusive **250,000-token automatic-compaction ceiling**. Models with smaller usable windows compact earlier to preserve their response reserve. The ceiling does **not** shrink the model's advertised context window or change its reasoning level.

The fork includes provider-native compaction and checkpoint replay for supported OpenAI/Codex routes, including Astra handling. Other routes use the supported summarization path unless a native-compaction endpoint has been explicitly verified. A shared model name does not imply that Azure, GitHub, Bedrock, or another gateway implements the same protocol.

Compaction work includes saved-checkpoint restoration, continuation of incomplete subagent work, and preservation of queued input. WebSocket handling retains connection reuse and incremental requests where supported, with bounded recovery that avoids transparently replaying a partially delivered response.

Fast/service-tier selection remains separate from reasoning effort and context management, and depends on the provider. It is not a prerequisite for using Astra.

### Memory that you can inspect and correct

Optimus provides a continual harness with project-scoped memory and explicit controls for recall, learning, provenance, and recovery.

- **Separate scopes:** retain session-specific facts, reusable project knowledge, and global harness entries.
- **Selective recall:** bring relevant entries into context within bounded recall limits.
- **Evidence references:** associate learned entries with the material used to produce them.
- **Corrections and history:** inspect revisions, supersede outdated facts, and roll back unchanged edits.
- **Selected history import:** review and recover knowledge from chosen saved sessions instead of indiscriminately importing every conversation.
- **Optional sharing:** explicitly share selected project memories through a configured service. Host-specific facts and full transcripts are not automatically shared.

Recall and learning are independent controls. You can stop new automatic learning while continuing to use existing memory, or turn off recall without deleting the underlying knowledge.

```text
/memory status
/memory search deployment decisions
/memory history
/memory recall off
/memory learning off
```

Current project memory is stored in versioned JSON state. Ordinary recall does not require a separate vector database or an extra model request. **TencentDB integration is a future direction; it is not a requirement for the memory system that ships today.**

See [Project memory](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/project-memory.md) for source validation, imports, backups, sharing, and the Python memory API.

### Native Windows as a real operating environment

Optimus carries Windows-focused work across process management, Python execution, persistence, and session recovery. The current Windows path runs on native Windows with Git Bash; WSL is not required for that path.

The work includes safer worker recovery, Windows path handling, bounded persistence retries, Python skill packaging, and kernel interruption and cleanup behavior. Session browsing distinguishes main conversations from inactive subagents, while retaining access to child histories through their parents.

The terminal is a client of the running agent services. A terminal disconnect and a worker failure are different events, and the harness tracks them separately. Persistent deployments must also launch those services independently of disposable editor or terminal-host processes.

See [Windows setup](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/windows.md). Machine-specific gateways, credentials, scheduled tasks, and guarded installation packages are deployment configuration—not bundled public defaults.

### Long-running work with explicit controls

Goals, autonomous continuation, heartbeats, and schedules support work that spans multiple turns or sessions. Background agents can be detached and revisited, with saved history and execution state available for recovery.

Use explicit budgets and stop controls. An idle worker is not proof of a completed task, and a successful tool call is not proof that the overall objective is finished. The fork adds bounded incomplete-child recovery and runtime-owned result delivery so parent agents are less dependent on a child remembering to send its final report.

### Multiple models, one harness

Optimus supports provider authentication, custom model definitions, and model-specific reasoning and tool behavior. Integrations include OpenAI/Codex, GitHub Copilot, Azure OpenAI/Foundry-compatible routes, Amazon Bedrock, and OpenAI-compatible services such as Ollama endpoints.

Use the model appropriate to each task: a planner or reviewer, an implementation agent, or a focused researcher. Authentication method, subscription entitlement, available models, image support, service tiers, and compaction capability vary by provider. Custom gateways require their own configuration.

See [Providers](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/providers.md) and [Custom models](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/models.md).

### Efficiency you can measure

The performance work targets both the host runtime and the conversation it manages:

- Asynchronous, ordered worker-state persistence reduces blocking filesystem work.
- Lossless tool-text deduplication reduces repeated text **on disk** without removing the visible tool result.
- Recent-first history loading reduces the amount of history transferred to the interface during supported reopen flows.
- Bounded retry behavior respects cancellation and avoids replaying visible partial output.
- Optional local metrics record request, tool, snapshot, compaction, retry, and reopen measurements where available.

**Detailed performance metrics are off by default and must be explicitly enabled before starting the agent.** There is currently no `/metrics` toggle. Set `PRIME_AGENT_PERFORMANCE_METRICS=1` in the environment inherited by the agent services; unset it or set it to `0` to disable collection on the next start.

Metrics normally go to `~/.prime/agent/performance-metrics`. Summarize a collected directory with:

```bash
node packages/coding-agent/scripts/summarize-performance-metrics.mjs --dir /path/to/performance-metrics
```

The recorder does not store prompt bodies, tool arguments, credentials, or reasoning text. Unavailable measurements remain unavailable rather than being reported as zero. Ordinary `/usage` totals are useful, but are not a substitute for this diagnostic breakdown.

Experimental model-facing output reduction, iterative-summary consolidation, and per-variable snapshot storage remain opt-in. Their presence in the code is not a claim of measured speed, quality, or token savings. See [Performance changes](https://github.com/telemusai/optimus-agent/blob/main/docs/performance-update.md) and [Metrics](https://github.com/telemusai/optimus-agent/blob/main/docs/performance-metrics.md).

## Jev integration

Optimus can put the same decision questions it is making to the TypeSafe System One ("Jev") service at `https://api.typesafe.ai/v1/systemone`, so an external recommendation can be compared against what the agent actually did.

There are two operative modes. **Compare** is a shadow mode: it never changes a run, recommendations are recorded for comparison only, and each answer is stored as not applied. **Active** applies an accepted answer to the next provider request, within a narrow reversible set (below). Jev is **off by default**, and `/jev` is the only command that sets the mode.

```text
/jev            # menu
/jev compare    # shadow mode for this chat (also: /jev on)
/jev active     # apply an accepted answer to the next provider request
/jev off        # no calls, no network, no overhead beyond the mode check
/jev status     # mode, the scope that decided it, and credential source presence
/jev key        # enter an API key
/jev key clear  # remove the saved key
```

An explicit per-session mode wins over the default for new chats, so `/jev off` in one chat does not turn Jev off elsewhere, and `/jev compare` in one chat is not cancelled by a global default of off. A key being present never enables Jev on its own. `on` is the short alias for Compare; only the explicit `/jev active` spelling selects the mode that may change a request, so nothing arms it by shorthand.

### Active mode: what gets applied

Active applies an accepted answer to the outgoing request body, and only these two effects exist:

| Category | Effect |
|---|---|
| `tool_requirement == "none"` | removes `tools` and `tool_choice` from that one request |
| `complexity` `low` or `high` | moves an already-present `reasoning_effort` one step |

Nothing else is appliable. Active never adds a key it did not find, never changes the model or provider, and never writes session thinking-level state. It cannot touch permissions, context, memory, compaction, continuation, subagents, agent messages, depth, concurrency or budgets.

Acceptance needs confidence of at least `0.7` and a decision no older than three seconds. A refused answer, a low-confidence answer, a missing confidence, a transport failure, a 2.5 s deadline, an open circuit breaker (three consecutive failures, 30 s cooldown) or a missing credential leaves the request unchanged and is recorded with a reason. Active decides at the provider-request boundary, so it adds latency to that turn instead of running in a background queue like Compare.

Compare mode asks bounded questions across eleven decision categories: task classification, complexity, tool requirement, tool candidates, subagent requirement, subagent model routing, context relevance, memory relevance, continue/stop/escalate, result sufficiency, and first-pass verification. Questions are grouped into bundles at defined lifecycle stages such as `turn_start`, `tool_call`, `agent_end`, and `model_select`, and each bundle carries a bounded state snapshot treated as untrusted data.

### What leaves the machine

Every operative mode sends the same summary of what the agent is doing to an external service, so the bridge keeps credential material and raw content out of the request:

- **Raw tool arguments are not observed.** The bridge records tool identity, a tool-call id, and an explicit `args_omitted` marker. Tool arguments can carry API keys, `Authorization` headers, and passwords, and the evaluators only need tool identity. A source guard fails the test suite if argument capture returns.
- **Bound before redacting.** Excerpts read at most the first 4096 characters of a source, so a multi-megabyte tool result is never fully copied just to keep a short excerpt.
- **Redact before the payload exists.** Credential-shaped material is replaced with `[redacted]` while the excerpt is built: authorization headers, provider key prefixes, private-key blocks, secret-named assignment values, URL userinfo, and JWT-shaped strings.

Redaction is pattern-based. It does not make arbitrary private task text safe to disclose, so treat an operative mode as an opt-in disclosure you control per session.

### Credentials, settings, and records

The credential is read from a saved credential first (the Windows DPAPI envelope at `<agent-dir>/jev/typesafe.jev-credential.json`), then from `TYPESAFE_API_KEY`, then from `JEV_API_KEY`. Windows uses the DPAPI store; macOS and Linux use the environment variables, because those builds have no DPAPI store. A hand-written `KEY=value` file at the envelope path is not read. `/jev status` reports which source is in use and never prints the secret.

Local state stays under `jev/` in the agent directory: `jev-settings.json` holds the mode default for new chats, per-session modes, and the key id, and `records.jsonl` holds the records (`jev.compare/1` rows for Compare, `jev.active/1` rows for Active). Set `JEV_BASE_URL` to point a run at a staging or replay endpoint instead of the production service.

See [Jev observation data minimization](docs/JEV_DATA_MINIMIZATION.md).

## Telegram access

Pair a private Telegram bot chat with an Optimus session to send messages, inspect context, change supported session settings, or stop work from your phone.

Start with `/telegram` in the terminal. Setup walks through BotFather, token entry, and a one-time pairing link. Once paired, use `/help` in the bot chat to see the available commands.

The current connector supports one paired private account and text messages. Your computer and agent services must remain running. Pairing grants control of the connected session and its tools: keep the token and pairing link private.

See [Telegram setup and recovery](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/telegram.md).

## The Rust migration

The Rust application is included on `main` alongside the TypeScript reference. Launch the installed application with `optimus-agent`; Cargo's internal executable remains `optimus-rust`. The `optimus-agent.sh` source launcher runs the TypeScript reference.

The Rust workspace is organized around four components:

| Crate | Responsibility |
| --- | --- |
| `pi-ai` | Model providers, streaming, authentication, and request/response types |
| `pi-agent-core` | Agent execution and tool-call orchestration |
| `pi-tui` | Terminal rendering and interaction |
| `pi-coding-agent` | Application runtime, sessions, daemon/workers, tools, and integrations |

The Python execution environment remains part of the design. Moving the harness to Rust does not mean removing Python tools or rewriting users' Python skills.

The objectives are clearer ownership, predictable concurrency, lower host overhead, and native deployment while preserving the existing behavioral contracts. Compilation alone does not establish parity: session recovery, compaction checkpoints, provider behavior, skills, and process cleanup still need runtime validation.

See [Rust validation and remaining gaps](docs/RUST_MAIN_READINESS.md) for the verified paths and known limitations. Use an isolated profile while evaluating the port. No blanket speedup or token-saving claim is made for the migration.

## Get started

### Rust implementation

For an installed Rust release, launch it from your project directory with:

```bash
optimus-agent
```

The maintained launcher is [`scripts/optimus-agent`](scripts/optimus-agent). See [the installation layout and launcher guide](docs/RUST_LAUNCHER.md). In the TUI, `/model` opens searchable model selection; `/model <search>` prefills the search or selects an exact, unambiguous reference.

Reopen the last saved conversation in the current project with:

```bash
optimus-agent --continue
```

Automatic continuation skips empty drafts. To choose a particular saved session, use `optimus-agent --resume /path/to/session.jsonl`.

From a checkout of this repository, build with Cargo:

```bash
cargo build --locked -p pi-coding-agent --bin optimus-rust
./target/debug/optimus-rust --help
./target/debug/optimus-rust --version
```

For a separate development profile on macOS/Linux:

```bash
mkdir -p .port-env/agent
PRIME_AGENT_CODING_AGENT_DIR="$PWD/.port-env/agent" \
  ./target/debug/optimus-rust --daemon-socket /tmp/optimus-dev-$UID.sock
```

Configure providers with `/login` and `/model`. Keep the checkout available for the Python runtime and bundled resources; copying the executable alone is not a complete installation. See the [validation notes](docs/RUST_MAIN_READINESS.md) for toolchain details and Windows validation limits.

### TypeScript reference implementation

Use Node.js 22.9 or newer and a compatible npm installation. The following source-launch commands run in Bash on macOS/Linux or Git Bash on Windows:

```bash
git clone https://github.com/telemusai/optimus-agent.git optimus-agent
cd optimus-agent
npm ci
./optimus-agent.sh
```

For a compiled run from the same checkout:

```bash
npm run build
./optimus-agent.sh --dist
```

The source launcher is `optimus-agent.sh`. Configuration paths retain `.prime/agent` for compatibility.

If you already use Optimus, select an isolated `PRIME_AGENT_CODING_AGENT_DIR` before experimenting with a different build. Existing `PRIME_AGENT_*` environment variables and `.prime` paths retain their names for configuration and session compatibility. Follow the [development guide](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/development.md) for profile isolation and checks, and the [Windows guide](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/windows.md) for shell setup.

On first launch, use `/login` to configure a provider, `/model` to select a model, and `/effort` to choose its supported reasoning level.

## Everyday commands

| Command | Purpose |
| --- | --- |
| `/login` | Configure provider authentication |
| `/model`, `/effort` | Select a model and supported reasoning level |
| `/new`, `/resume`, `/name` | Start, reopen, or name a session |
| `/context`, `/usage` | Inspect context and recorded token usage |
| `/compact` | Request context compaction |
| `/memory` | Inspect and manage memory |
| `/refine` | Refine reusable harness instructions and knowledge |
| `/goal`, `/autonomous` | Manage objectives and autonomous continuation |
| `/heartbeat` | Configure recurring session prompts |
| `/tree`, `/fork`, `/clone` | Navigate or branch session history |
| `/telegram` | Manage the Telegram connection |
| `/jev` | Configure Jev mode (off/compare/active) and its credential |
| `/settings`, `/mcp`, `/reload` | Configure and reload integrations and resources |

## Safety and compatibility

Optimus executes model-generated code and commands with the permissions of its host user. **A separate worker or Python process is not a security sandbox.** Use an external sandbox or restricted account for untrusted work, review changes, and keep recoverable checkpoints.

Preserve sessions, memory, configuration, and credentials when upgrading. New transcript encodings or checkpoint formats can require their matching readers; switching to an older executable is not automatically a safe rollback. Do not run competing builds against the same active profile.

## Documentation

- [Documentation index](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/index.md)
- [Architecture](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/architecture.md)
- [RLM workflows](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/rlm.md) and [Skills](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/skills.md)
- [Background agents](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/long-running-agents.md)
- [Project memory](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/project-memory.md)
- [MCP integrations](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/mcp-integrations.md)
- [Providers and authentication](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/providers.md)
- [Settings](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/settings.md)
- [Performance metrics](https://github.com/telemusai/optimus-agent/blob/main/docs/performance-metrics.md)
- [Development](https://github.com/telemusai/optimus-agent/blob/main/packages/coding-agent/docs/development.md)

Some linked documentation retains upstream terminology. The source of truth for Optimus development is [telemusai/optimus-agent](https://github.com/telemusai/optimus-agent).

## Contributing

Keep changes focused, describe the behavior they alter, and test the affected paths. Preserve provider-specific contracts, memory scopes, session compatibility, and Windows behavior. Performance changes should include measurements; Rust contributions should demonstrate behavior, not just successful compilation.

See [AGENTS.md](https://github.com/telemusai/optimus-agent/blob/main/AGENTS.md) and [CONTRIBUTING.md](https://github.com/telemusai/optimus-agent/blob/main/CONTRIBUTING.md) for repository conventions.

## Foundations and acknowledgements

Optimus builds on [Prime Agent](https://github.com/PrimeIntellect-ai/prime-agent) and [Pi](https://github.com/earendil-works/pi). We thank the Prime Intellect team, Mario Zechner, and the contributors to both projects for the agent, RLM, continual-harness, provider, and terminal foundations.

The project retains that lineage while developing its own runtime, integrations, and operating model. Original copyright notices are preserved.

## License

[MIT](https://github.com/telemusai/optimus-agent/blob/main/LICENSE).
