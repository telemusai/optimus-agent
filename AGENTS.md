# Development Rules

## Conversational Style

- No fluff or cheerful filler text
- Keep answers short and concise
- No emojis in commits, issues, PR comments, or code
- Technical prose only, be kind but direct (e.g., "Thanks @user" not "Thanks so much @user!")

## Code Quality

- Read files in full before making wide-ranging changes, before editing files you have not already fully inspected, and when the user asks you to investigate or audit something. Do not rely only on search snippets for broad changes.
- Don't be too verbose with comments in the code. Only write comments when there is serious ambiguity
- Inspect Cargo dependencies and existing Rust interfaces before adding new APIs.
- NEVER remove or downgrade code to fix type errors from outdated dependencies; upgrade the dependency instead
- Always ask before removing functionality or code that appears to be intentional
- Do not preserve backward compatibility unless the user explicitly asks for it
- Never hardcode key checks with, eg. `matchesKey(keyData, "ctrl+x")`. All keybindings must be configurable. Add default to matching object (`DEFAULT_EDITOR_KEYBINDINGS` or `DEFAULT_APP_KEYBINDINGS`)

## Commands

- After code changes, run `bash scripts/check.sh`: whitespace, Cargo workspace/all-target checks, packaging tests, and native resource layout validation.
- Run focused Cargo tests for changed behavior. `./test.sh` runs workspace unit tests and the Python suites; use `--test <name>` for integration tests.
- If you create or modify a test file, run it and iterate until it passes.
- Use isolated temporary profiles, local provider fixtures, and synthetic credentials. Never use real provider APIs, keys, or paid tokens for regression tests.
- Fix errors and any warnings introduced by the change; report unrelated existing failures.
- The application is Rust-only. Do not reintroduce the TypeScript implementation or a root npm workspace.

## Daemon Protocol Changes

- Classify every daemon command, event, and response-shape change as backward-compatible, capability-gated, or incompatible.
- Add optional features behind a negotiated server capability. Clients must check the capability before sending the command or depending on the event.
- Bump `DAEMON_PROTOCOL_VERSION` for incompatible changes or when startup begins requiring behavior an older daemon cannot provide.
- Update `DAEMON_SCHEMA_REVISION`, the command/event compatibility maps, and both new-client/old-daemon and old-client/new-daemon tests for every wire change.
- Optional daemon metadata and UI features must degrade locally. They must not prevent the agent, session attachment, or interactive startup from working.
- Never make a new daemon command part of startup without a protocol or capability gate.

## Dependencies

- A 7-day minimum release age applies to routine dependency updates. `.github/dependabot.yml` configures the matching cooldown for Cargo and uv.
- Keep `Cargo.lock` and `prime-agent-runtime/uv.lock` checked in; use locked builds.
- Document any urgent security exception to the release-age policy.

## GitHub Workflow

When creating issues:

- Add `pkg:*` labels to indicate which package(s) the issue affects
  - Available labels: `pkg:agent`, `pkg:ai`, `pkg:coding-agent`, `pkg:tui`
- If an issue spans multiple packages, add all relevant labels

When posting issue/PR comments:

- Write the full comment to a temp file and use `gh issue comment --body-file` or `gh pr comment --body-file`
- Never pass multi-line markdown directly via `--body` in shell commands
- Preview the exact comment text before posting
- Post exactly one final comment unless the user explicitly asks for multiple comments
- If a comment is malformed, delete it immediately, then post one corrected comment
- Keep comments concise, technical, and in the user's tone

When closing issues via commit:

- Include `fixes #<number>` or `closes #<number>` in the commit message
- This automatically closes the issue when the commit is merged

## PR Workflow

- Analyze PRs without pulling locally first
- If the user approves: create a feature branch, pull PR, rebase on main, apply adjustments, commit, merge into main, push, close PR, and leave a comment in the user's tone
- We work in feature branches until everything is according to the user's requirements. Never merge PRs by yourself.

## Testing Prime Agent Interactive Mode with tmux

To test Prime Agent's TUI in a controlled terminal environment:

```bash
# Create tmux session with specific dimensions
tmux new-session -d -s prime-agent-test -x 80 -y 24

# Start Prime Agent from source
tmux send-keys -t prime-agent-test "cd /path/to/optimus-agent && ./optimus-agent.sh" Enter

# Wait for startup, then capture output
sleep 3 && tmux capture-pane -t prime-agent-test -p

# Send input
tmux send-keys -t prime-agent-test "your prompt here" Enter

# Send special keys
tmux send-keys -t prime-agent-test Escape
tmux send-keys -t prime-agent-test C-o  # ctrl+o

# Cleanup
tmux kill-session -t prime-agent-test
```

You, yourself, are often running into a tmux session, so be careful when killing tmux sessions. Lots of other processes can be running on different tmux sessions/

## Changelog

- Add `.changes/<slug>.md` for user-visible changes, one fragment per PR.
- Each fragment contains plain bullet lines beginning with Added, Changed, Fixed, or Removed.
- Do not modify historical released entries in `resources/agent/CHANGELOG.md`.
- Purely internal changes may opt out via the `no-changelog` PR label.

## Adding a New LLM Provider

- Extend the shared types and provider implementation in `crates/pi-ai/`.
- Register credentials, model resolution, login guidance, and CLI help through the existing Rust interfaces.
- Add offline provider fixtures covering streaming, cancellation, usage, tool calls, and errors.
- Document setup in `resources/agent/docs/providers.md` and add a changelog fragment.

## Releasing

- Keep the workspace version in `Cargo.toml` and the resource identity in `resources/agent/package.json` aligned.
- Run the native checks and relevant tests before building `optimus-rust` with Cargo.
- `python3 scripts/rust_release.py stage --binary <path> --output <directory>` creates a portable archive and checksum.
- The Build binaries workflow builds native Linux, macOS, and Windows bundles. Version tags publish GitHub Release assets; manual runs upload reviewable artifacts.
- Release tags and publishing require user authorization. There is no npm release process.

## **CRITICAL** Git Rules for Parallel Agents **CRITICAL**

Multiple agents may work on different files in the same worktree simultaneously. You MUST follow these rules:

### Committing

- **ONLY commit files YOU changed in THIS session**
- ALWAYS include `fixes #<number>` or `closes #<number>` in the commit message when there is a related issue or PR
- NEVER use `git add -A` or `git add .` - these sweep up changes from other agents
- ALWAYS use `git add <specific-file-paths>` listing only files you modified
- Before committing, run `git status` and verify you are only staging YOUR files
- Track which files you created/modified/deleted during the session

### Forbidden Git Operations

These commands can destroy other agents' work:

- `git reset --hard` - destroys uncommitted changes
- `git checkout .` - destroys uncommitted changes
- `git clean -fd` - deletes untracked files
- `git stash` - stashes ALL changes including other agents' work
- `git add -A` / `git add .` - stages other agents' uncommitted work
- `git commit --no-verify` - bypasses required checks and is never allowed

### Safe Workflow

```bash
# 1. Check status first
git status

# 2. Add ONLY your specific files
git add crates/pi-ai/src/providers/transform_messages.rs
git add .changes/eng-1234-fix-resize.md

# 3. Commit
git commit -m "fix(ai): description"

# 4. Push (pull --rebase if needed, but NEVER reset/checkout)
git pull --rebase && git push
```

### If Rebase Conflicts Occur

- Resolve conflicts in YOUR files only
- If conflict is in a file you didn't modify, abort and ask the user
- NEVER force push

### User override

If the user instructions conflict with rules set out here, ask for confirmation that they want to override the rules. Only then execute their instructions.
