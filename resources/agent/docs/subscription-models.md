# Subscription model metadata

`crates/pi-ai/src/models.subscription.json` adds reviewed models to the native
catalog. It does not change sign-in, select a model automatically, or grant an
account access. Explicit user model definitions still override built-in entries.

## GPT-6 Sol and Luna on Codex

Selectors: `openai-codex/gpt-6-sol` and `openai-codex/gpt-6-luna`.
They use the existing Codex sign-in and `openai-codex-responses` transport.

Reviewed sources:

- [Codex CLI rust-v0.156.1 catalog](https://github.com/openai/codex/blob/rust-v0.156.1/codex-rs/models-manager/models.json)
  pins the exact IDs, text/image input, a 272,000-token default context, and
  reasoning levels `low`, `medium`, `high`, `xhigh`, and `max` for both models.
  Its raw SHA-256 is
  `1892a933a420e79a30779ef5f1ce5b7dbc145464aa6fb1432106a9e9859981d3`.
- [Sol model details](https://developers.openai.com/api/docs/models/gpt-6-sol)
  and [Luna model details](https://developers.openai.com/api/docs/models/gpt-6-luna)
  publish a 128,000-token output limit. Their larger API context is not substituted
  for the Codex subscription catalog's default context.
- [OpenAI pricing](https://platform.openai.com/docs/pricing) supplies the standard
  API per-million-token estimates used in the catalog: Sol input $2/output $10,
  cache read $0.20/cache write $2.50; Luna input $0.10/output $0.50,
  cache read $0.01/cache write $0.125. These estimates do not measure subscription
  credits, account charges, or long-context pricing tiers.

The discovery client version is 0.156.1, the published CLI release that bundled
these models. Its source declares minimum client version 0.155.0. The existing
account-scoped Codex catalog filter is retained. Missing IDs or failed catalog
reads do not trigger a substitute model. Static registration and offline fixtures
are not proof of an account's current rollout, policy, quota, or entitlement.

Sol's documented `ultra` mode includes automatic delegation and is not exposed
as an Optimus thinking level. `off` and `minimal` are not advertised for these
models. Existing tier and fast-mode policy is unchanged.

## GPT-6 Sol and Luna on GitHub Copilot

Selectors: `github-copilot/gpt-6-sol` and `github-copilot/gpt-6-luna`.
They use the existing Copilot sign-in and `openai-responses` transport. Existing
OAuth token routing still selects the account's Copilot base URL.

A single authorized provider `/models` read on 2026-09-23 verified both exact IDs
and `/responses` support. The selected metadata is retained in the offline fixture
`crates/pi-ai/tests/fixtures/copilot-sol-luna-catalog.json`; the sanitized receipt
SHA-256 is `b9d3711c8860cd90b1658a2a7a647be446b2e7ec5d35a7c58415e9a2c4794a30`.
Both entries use the provider's limits, not the larger public API figures:

- 400,000 total context tokens, with a separate 272,000-token input ceiling.
- 128,000 maximum output tokens.
- Text/image input, tool calls, parallel tool calls, and streaming.
- Reasoning `none`, `low`, `medium`, `high`, `xhigh`, and `max`. Optimus `off`
  maps explicitly to `none`; `minimal` is not advertised. Omitted reasoning keeps
  the existing provider default. Copilot requests still omit `service_tier`.

The standard API cost estimates above are retained as estimates, not subscription
billing guarantees. The sampled account reported enabled policy and model-picker
availability. This does not grant another account access or prove successful
inference, current quota, or future rollout. Copilot provider errors remain visible;
no automatic model substitution or policy-enabling call was added.
