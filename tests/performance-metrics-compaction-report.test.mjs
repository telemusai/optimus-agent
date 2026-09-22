import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

test("reports compaction phases and keeps summary attempts separate from ordinary requests", () => {
	const directory = mkdtempSync(join(tmpdir(), "compaction-metrics-report-"));
	try {
		const record = (operation, component, outcome, measurements) => ({
			schemaVersion: 1, operation, outcome, measurements,
			correlation: { sessionId: "isolated" },
			identity: { component, provider: "faux", model: "fixture", api: "faux" },
		});
		const rows = [
			record("compaction_history", "compaction", undefined),
			record("provider_attempt", "compaction", undefined, { attempt_ordinal: 1 }),
			record("provider_attempt", "compaction", "started", { attempt_ordinal: 1 }),
			record("provider_attempt", "compaction", "success", { total_ms: 600, dispatch_to_network_terminal_ms: 590, attempt_ordinal: 1 }),
			record("compaction_history", "compaction", "success", { total_ms: 610 }),
			record("provider_attempt", "provider", "success", { total_ms: 30 }),
			record("session_input", "session", "success", { queue_ms: 50, input_agent_message: 1 }),
		];
		writeFileSync(join(directory, "performance-v1-isolated.jsonl"), rows.map((row) => JSON.stringify(row)).join("\n") + "\n");
		const report = JSON.parse(execFileSync(process.execPath, [fileURLToPath(new URL("../scripts/summarize-performance-metrics.mjs", import.meta.url)), "--dir", directory, "--json"], { encoding: "utf8" }));
		assert.equal(report.records.invalid, 0);
		assert.equal(report.records.startedByOperation.provider_attempt, 2);
		assert.equal(report.measurements.byOperation.provider_attempt.attempt_ordinal.available, 1);
		assert.equal(report.measurements.byOperation.provider_attempt.attempt_ordinal.available, 1);
		assert.equal(Object.values(report.measurements.providerAttemptsByIdentity)[0].total_ms.mean, 30);
		assert.equal(Object.values(report.measurements.compactionAttemptsByIdentity)[0].total_ms.mean, 600);
		assert.equal(Object.values(report.measurements.compactionAttemptsByIdentity)[0].dispatch_to_network_terminal_ms.mean, 590);
		assert.equal(report.measurements.byOperation.session_input.input_agent_message.sum, 1);
	} finally {
		rmSync(directory, { recursive: true, force: true });
	}
});
