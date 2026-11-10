# Quickstart

This page gets you from install to a useful first session with Optimus by Telemus AI.

## Install

Install the prerequisites listed in the [root README](../../../README.md#install-and-update), then install the latest stable release on Linux or macOS:

```bash
curl -fsSL https://telemus.ai/optimus-agent/install.sh | sh
```

The installer selects the highest stable release tag from `telemusai/optimus-agent`, builds it locally with locked Cargo dependencies, and installs the `optimus-agent` command. It does not install an untagged `main` build. See the [installation guide](../../../docs/RUST_LAUNCHER.md) for Windows and installation layout details.

Then start Optimus in the project directory you want it to work on:

```bash
cd /path/to/project
optimus-agent
```

To run a source checkout instead, install Rust/Cargo 1.98.0 or newer, Python 3.11 or newer, `uv`, and a Bash shell:

```bash
git clone https://github.com/telemusai/optimus-agent.git
cd optimus-agent
cargo build --locked -p pi-coding-agent --bin optimus-rust
./optimus-agent.sh
```

The source runner preserves the directory from which it is invoked, so you can also call `/path/to/optimus-agent/optimus-agent.sh` from another project. Launching Optimus does not require Node or npm.

The installed launcher uses `~/.config/optimus-rust/` for settings, credentials, and sessions. The source runner defaults to the inherited `~/.prime/agent/` profile. Set `PRIME_AGENT_CODING_AGENT_DIR` to choose a different profile; the examples below use the installed profile.

## Authenticate

Optimus can use subscription providers through `/login`, or API-key providers through environment variables or its auth file.

### Option 1: Subscription Login

Start Optimus and run:

```text
/login
```

Then select a provider. Built-in subscription logins include Claude Pro/Max, ChatGPT Plus/Pro (Codex), and GitHub Copilot.

### Option 2: API Key

Set an API key before launching Optimus:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
optimus-agent
```

You can also run `/login` and select an API-key provider to store the key in `~/.config/optimus-rust/auth.json`.

See [Providers](providers.md) for all supported providers, environment variables, and cloud-provider setup.

## First Session

Once Optimus starts, type a request and press Enter:

```text
Summarize this repository and tell me how to run its checks.
```

In IPython mode, Optimus gives the model a persistent `ipython` tool. The long-lived kernel is a control environment for reading and editing files, running project commands, inspecting data, retaining Python state, and invoking installed skills. The kernel runtime is bootstrapped automatically on first use; set `PRIME_AGENT_KERNEL_PYTHON` to use an existing Python environment with `prime-agent-runtime`. Use `/mode` to choose another supported execution mode.

Optimus runs in your current working directory and can modify files there. Use git or another checkpointing workflow if you want easy rollback.

## Recursive Subagents

Recursive subagents are a built-in Optimus capability. The model spawns independent work from the Python REPL with `await rlm("subtask")`; each call returns at admission with a child handle and never returns the answer. Children send requested results as explicit `agent_message` replies to the parent or write them to files. Child agents use the same native agent runtime, providers, tools, skills, and session machinery as the parent.

You can prompt the model to use that capability directly:

```text
Review authentication and test coverage as independent subtasks. Run them in parallel, then synthesize the findings.
```

See [RLM Runtime Architecture](rlm-runtime.md) for the API and execution model.

## Give Optimus Project Instructions

Optimus loads context files at startup. Add an `AGENTS.md` file to tell it how to work in a project:

```markdown
# Project Instructions

- Run `npm run check` after code changes.
- Do not run production migrations locally.
- Keep responses concise.
```

Optimus loads:

- `~/.config/optimus-rust/AGENTS.md` for global instructions in the installed profile
- `AGENTS.md` or `CLAUDE.md` from parent directories and the current directory

Restart Optimus, or run `/reload`, after changing context files.

## Common Things to Try

### Reference Files

Type `@` in the editor to fuzzy-search files, or pass files on the command line:

```bash
optimus-agent @README.md "Summarize this"
optimus-agent @src/app.ts @src/app.test.ts "Review these together"
```

Images can be pasted with Ctrl+V (Alt+V on Windows) or dragged into supported terminals.

### Run Shell Commands

In interactive mode:

```text
!npm run lint
```

The command output is sent to the model. Use `!!command` to run a command without adding its output to model context. During agent work, the model normally runs project commands from the Python REPL with `bash()`.

### Switch Models

Use `/model` or Ctrl+L to choose a model. Use `/effort` to set the reasoning level. Use Ctrl+P / Shift+Ctrl+P to cycle through scoped models.

### Continue Later

Sessions are saved automatically under `~/.config/optimus-rust/sessions/` in the installed profile:

```bash
optimus-agent -c                  # Continue the most recent session
optimus-agent -r [path|id]        # Browse sessions or open a specific session
```

Inside Optimus, use `/resume`, `/new`, `/tree`, `/fork`, and `/clone` to manage sessions. Persistent sessions run in worker processes, so closing the TUI detaches from the agent rather than necessarily stopping it. Use `optimus-agent agents` to inspect or reattach to active work.

### Non-Interactive Mode

For one-shot prompts:

```bash
optimus-agent -p "Summarize this codebase"
cat README.md | optimus-agent -p "Summarize this text"
optimus-agent -p @screenshot.png "What's in this image?"
```

Use `--mode json` for JSON event output or `--mode rpc` for process integration.

## Next Steps

- [Using Optimus](usage.md) - interactive mode, slash commands, sessions, context files, and CLI reference.
- [Providers](providers.md) - authentication and model setup.
- [Settings](settings.md) - global and project configuration.
- [Keybindings](keybindings.md) - shortcuts and customization.
- [Optimus Packages](packages.md) - install shared extensions, skills, prompts, and themes.

Platform notes: [Windows](windows.md), [Termux](termux.md), [tmux](tmux.md), [Terminal setup](terminal-setup.md), [Shell aliases](shell-aliases.md).
