# Gateway credential-helper warning — mandatory wrap-up investigation

Status: OPEN. User reported a fresh warning while the repair/Jev teams were working, then reported the agent continuing. This is separate from the compaction incident.

## Evidence collected read-only on 2026-09-19 around 09:52–09:56 UTC
- Screenshot: C:/Users/OPENCL~1/AppData/Local/Temp/codex-clipboard-7daf58cb-bdf3-4255-be4c-ac2c232c4dd8.png.
- Exact displayed failure: Failed to resolve API key for provider "ollama-cloud" from shell command: powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:/Users/openclawuser/optimus-rust-test-20260915/cloud-model-gateway/ensure-gateway.ps1.
- Screenshot shows depth-0 Ollama Cloud GLM 5.3 Flash and six children; the repair coordinator is the likely matching chat, but the warning itself was not found in bounded tails of either coordinator transcript. Its precise originating invocation, exit status and time remain unresolved.
- Read-only get_state returned both repair (7c61a488ca86) and Jev (dd552b3c2d08) coordinators working, streaming and running tools, using ollama-cloud/glm-5.3-flash.
- Existing gateway.log contained successful HTTP 200 requests for both GLM and DeepSeek at 09:53:07–09:53:15. These establish ongoing gateway traffic, not exact attribution of the failed helper call.
- Source inspected: source/crates/pi-coding-agent/src/core/resolve_config_value.rs and model_registry.rs. git diff --quiet against base cad234a17589f5838407421fb435bd74d5ceda76 confirmed both files unchanged at inspection.
- Installed helper inspected as source only: C:/Users/openclawuser/optimus-rust-test-20260915/cloud-model-gateway/ensure-gateway.ps1. It was NOT executed. No credential file was read or copied.

## Confirmed source behaviour
1. ensure-gateway.ps1 reads a local gateway credential, checks localhost gateway health/build identity, and outputs the credential only after health succeeds. It does not fetch the Ollama account's upstream API key. The UI wording alone is not evidence that the user's Ollama key is invalid.
2. It uses one-second health requests and two delayed retries (400/800 ms). If the task is already Running but health fails, it refuses to restart it; retain that protection. Cold startup can additionally wait eight seconds, and a final extra health check can fail after an earlier successful check.
3. The resolver has a ten-second command deadline (resolve_config_value.rs:20), turns launch failure, nonzero exit, timeout and empty output into None (configured_shell_result), and discards their distinctions. resolve_config_value_or_throw (around line 357) then emits the same generic warning plus command.
4. model_registry.rs:get_api_key_and_headers calls that uncached resolver for configured provider commands. Investigate repeated/concurrent helper invocations and their impact; do not assume the stored-auth cache covers this route.
5. The helper's possible cold-start duration can exceed its caller's ten-second budget. This is a source-supported deadline mismatch, NOT proof of this particular incident's cause.

## Required wrap-up work
- Trace/reproduce the relevant helper failure with isolated fake credentials, health endpoints and task/process abstractions. Do not run the production helper, change login, or restart the live gateway to investigate.
- Preserve structured, redacted reason/exit/elapsed data so timeout, launch failure, unhealthy service, wrong build and empty output are distinguishable. Never log stdout, tokens, raw auth or unfiltered secret-bearing command arguments/stderr.
- Prove bounded, consistent caller/helper deadlines and sensible recovery after a transient health failure; preserve wrong-build/unauthorised-response refusal. Do not simply suppress the warning, disable health/auth checks, enlarge arbitrary timeouts or retry indefinitely.
- Evaluate coalescing repeated concurrent credential resolutions only if justified by evidence; never retain invalid credentials indefinitely or silently ignore explicit auth rejection.
- Test transient failure then success, persistent failure, missing credential, wrong build, helper timeout, cancellation, concurrent resolution, no secret leakage, no unrelated process kill/restart, and preserved pending input.
- Decide and record whether an already-running request continued, a later resolution recovered, or only a stale UI warning remained. Current evidence does not distinguish these.
- Keep unresolved until the integrated candidate has an evidence-backed correction and regression proof. Codex independently verifies after the teams finish.

No production changes, helper execution, new provider requests, agent messages, restarts, cutover or scheduled monitoring were performed for this check.
