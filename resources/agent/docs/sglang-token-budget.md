# SGLang final-request token budgeting

For an existing `openai-completions` model served by a compatible SGLang endpoint, set the model compatibility field:

```json
{"compat":{"sglangTokenBudget":true}}
```

This is opt-in. It does not add a provider, proxy, server setting, or endpoint-wide concurrency limit. Other models do not make preflight requests.

## Request behavior

After payload hooks, Optimus counts the final prompt before generation. Text, tool definitions, tool history, and replayed reasoning use the same endpoint's `/tokenize` chat renderer. Images use `/chat/completions` with `stream:false,max_tokens:0,return_prompt_token_ids:true` and count the returned expanded prompt token IDs. Each generation therefore adds one HTTP preflight and its latency.

Count requests keep all prompt-affecting fields. They remove streaming options and normalize count-only output controls to `max_tokens:0`. The final generation request keeps its normal streaming controls. Optimus requires zero generated tokens and no returned content from an image count prefill. Returned raw prompt IDs must be nonempty integers and consistent across choices; missing or malformed IDs fail closed. IDs are counted in memory, never logged or added to the final generation request.

Raw model-input counts can exceed billed prompt usage. SGLang's K3 API excludes a three-token assistant generation stub from reported usage, while the model input and returned prompt IDs include it. The guard uses full input IDs, not billed usage or a guessed adjustment based on a model name. Tests must preserve this distinction when comparing usage with preflight counts.

The final output limit is `min(requested output, model.maxTokens, contextWindow - prompt tokens - 1)`. Text counting also respects a smaller tokenizer-reported context limit. If no positive output room remains, or exact counting fails, the request fails before generation. Messages are never truncated by this guard.

This guard does not change the normal output-default setting, global compaction threshold, compaction reserve, or input-limit setting. A large context window does not bypass the normal compaction policy.

## Requirements and limits

- The endpoint must support chat `/tokenize`, including tools and reasoning fields. Image counting requires zero-output completion support and returned prompt token IDs captured after multimodal expansion (`return_prompt_token_ids:true`). Billed usage alone is insufficient.
- Native inline image data URLs are supported. Mutable remote image URLs and unsupported audio/video blocks fail closed because a second fetch or media processor cannot be counted reliably with this contract.
- The serving model, tokenizer, chat template, and image processing must stay stable between count and generation. Counting is not a server-side atomic reservation. Revalidate this contract after a server/template upgrade.
- Count requests use existing authentication to the same base URL and never follow redirects. They use the normal configured request timeout (default10minutes), applied to the complete finite count response. The count response limit is16MiB. Generation retains its existing header-only timeout and SSE cancellation behavior.
- Server queue limits remain server settings. This feature does not add a shared request semaphore across sessions or subagents.

## Optional test audit

Set `OPTIMUS_K3_REQUEST_AUDIT_PATH` to an owned JSONL file path. Only models with `sglangTokenBudget:true` emit records. Audit failures stop generation; leaving the environment variable unset preserves normal no-audit behavior.

The `optimus.sglang.dispatch-audit.v1` record describes the **prepared HTTP request**, not a server receipt. It hashes the actual immutable serialized reqwest body immediately before execution. It includes provider/model, reasoning-effort enum, exact prompt count/method, requested/final output limits, context limit, count latency, message-role order, content/reasoning byte lengths and hashes, tool names/ID hashes/semantic argument hashes, and image counts/encoded lengths.

The audit never stores headers, credentials, raw prompts, raw reasoning, raw tool arguments/results, or image data. Combine it with native response/usage evidence to verify live behavior. Tool argument objects are sorted recursively before hashing. Tool IDs and result IDs use matching UTF8 SHA256 hashes.

Use a dedicated test-owned output path. Each record uses one append operation and an in-process mutex; readers must verify complete JSON records when collecting multiple processes. Do not treat the file as transactional cross-process logging. Protect audit files because hashes, lengths, tool names, and usage still reveal metadata.
