# Claude compaction recovery

Summary transcripts exclude internal thinking blocks, including signed and
redacted blocks, instead of flattening them into `[Assistant thinking]` text.
Visible assistant responses, user requests, tool calls and tool results remain
available to the summarizer. This applies to both serialization paths and to
conversation, split-turn and branch summaries; the stored conversation is not
rewritten. The handoff instructions focus on observable work and decisions.

A local HTTP trace of Claude Code 2.1.310 confirmed that its compaction request
keeps thinking in typed assistant blocks rather than promoting it into ordinary
user text. Optimus follows the same separation by omitting thinking from its
bounded text summary input. It retains its own handoff schema and chunking;
the complete request is not a byte-for-byte copy of Claude Code's request.
Anthropic's `reasoning_extraction` category has no recommended alternate model:
correct the request instead of attempting to route around that refusal.

Direct Anthropic summary requests for Opus 5.5, Opus 5, Fable 5/5.1 and
Sonnet 5.5 enable Anthropic's `fallbacks: "default"` routing with the
`server-side-fallback-2026-07-01` beta. Anthropic chooses an allowed alternate
Claude model when its classifier refuses the requested model's summary.
The session's chat model and credentials are unchanged. The option is attached
to each summary request, including split-turn prefixes and output-budget retries.

This follows [Anthropic's refusal recovery documentation](https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback).
It does not choose Astra or send conversation data to another provider. Normal
chat, Kiro, and older Claude model requests retain their existing behavior.

The daemon records a serving-model transition when Anthropic selects another
Claude. Summary token totals include all reported attempts; known models use
their individual prices. If the server selects a model absent from the local
catalog, its token counts remain included and a warning identifies that its
cost is unavailable.

If the API still refuses the summary, Optimus preserves the existing conversation,
reports the bounded category/explanation and request ID, and classifies the
failure as filtered for the existing recovery backoff. Refusal text and partial
summaries are never committed as checkpoints. Authentication, rate-limit and
transport failures retain their existing handling. Categories for which Anthropic
offers no alternate model can still refuse; this is not a guarantee that every
summary will complete.

The change runs in shared native compaction and therefore applies to Optimus TUI
and Telegram sessions. Telemus 3D/Web surfaces use a separate application and are
unaffected. No daemon command, event, response field, or capability changes:
existing diagnostic strings and usage fields carry the additional evidence.

Validation uses local HTTP/SSE fixtures and synthetic OAuth credentials, covering
summary routing, OAuth header preservation, terminal refusal classification,
serving model and usage, and unchanged chat/Kiro behavior.
