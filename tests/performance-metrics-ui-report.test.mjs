import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { mkdtempSync, rmSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { test } from "node:test";
import { fileURLToPath } from "node:url";

test("reports content-free client spans separately and keeps render aggregate counts", () => {
	const directory = mkdtempSync(join(tmpdir(), "ui-metrics-report-"));
	try {
		const rows = ["ui_input", "ui_input_ack", "ui_menu_open", "ui_session_open", "ui_render"].map((operation) => ({
			schemaVersion: 1, operation, outcome: "success", correlation: { sessionId: "isolated" },
			identity: { component: "session" },
			measurements: operation === "ui_render" ? { total_ms: 10, max_ms: 6, frame_count: 2 } : { total_ms: 7 },
		}));
		rows.push({ schemaVersion: 1, operation: "snapshot", outcome: "success", correlation: { sessionId: "isolated" }, identity: { component: "snapshot" }, measurements: { total_ms: 120, serialization_max_variable_ms: 80, serialization_slow_variables: 2, serialization_saved_ms: 15, serialization_skipped_ms: 5 } });
		writeFileSync(join(directory, "performance-v1-isolated.jsonl"), rows.map((row) => JSON.stringify(row)).join("\n") + "\n");
		const report = JSON.parse(execFileSync(process.execPath, [fileURLToPath(new URL("../scripts/summarize-performance-metrics.mjs", import.meta.url)), "--dir", directory, "--json"], { encoding: "utf8" }));
		assert.equal(report.records.invalid, 0);
		assert.equal(report.records.byOperation.ui_input_ack, 1);
		assert.equal(report.measurements.byOperation.ui_render.frame_count.sum, 2);
		assert.equal(report.measurements.byOperation.ui_render.max_ms.max, 6);
		assert.equal(report.measurements.byOperation.ui_menu_open.total_ms.mean, 7);
		assert.equal(report.measurements.byOperation.snapshot.serialization_max_variable_ms.max, 80);
		assert.equal(report.measurements.byOperation.snapshot.serialization_slow_variables.sum, 2);
		assert.equal(report.measurements.byOperation.snapshot.serialization_saved_ms.mean, 15);
		assert.equal(report.measurements.byOperation.snapshot.serialization_skipped_ms.mean, 5);
		assert.ok(report.notes.some((note) => note.includes("not durable delivery or model completion")));
	} finally {
		rmSync(directory, { recursive: true, force: true });
	}
});
