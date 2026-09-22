#!/usr/bin/env node

import { createReadStream } from "node:fs";
import { readdir, stat } from "node:fs/promises";
import { resolve } from "node:path";
import { StringDecoder } from "node:string_decoder";

const MAX_DEFAULT_FILES = 256;
const MAX_DEFAULT_BYTES = 128 * 1024 * 1024;
const MAX_DEFAULT_LINE_BYTES = 32 * 1024;
const SESSION_SET_LIMIT = 100_000;
const IDENTITY_GROUP_LIMIT = 128;
const OPERATIONS = new Set([
	"logical_request",
	"provider_attempt",
	"tool",
	"snapshot",
	"compaction",
	"compaction_prepare",
	"compaction_history",
	"compaction_prefix",
	"compaction_native",
	"compaction_persist",
	"compaction_restore",
	"file_retry",
	"session_reopen",
	"session_input",
	"ui_input",
	"ui_input_ack",
	"ui_render",
	"ui_menu_open",
	"ui_session_open",
	"recorder",
]);
const MEASUREMENTS = new Set([
	"total_ms",
	"wait_ms",
	"dispatch_to_response_headers_ms",
	"transport_open_ack_ms",
	"dispatch_to_first_event_ms",
	"dispatch_to_first_visible_ms",
	"dispatch_to_first_raw_ms",
	"dispatch_to_first_thinking_ms",
	"dispatch_to_first_tool_ms",
	"dispatch_to_first_text_ms",
	"dispatch_to_network_terminal_ms",
	"local_drain_ms",
	"transport_websocket",
	"local_gateway_wait_ms",
	"upstream_wait_ms",
	"serialization_ms",
	"serialization_cpu_ms",
	"serialization_max_variable_ms",
	"serialization_slow_variables",
	"serialization_saved_ms",
	"serialization_skipped_ms",
	"write_ms",
	"queue_ms",
	"next_cell_delay_ms",
	"reopen_ms",
	"serialized_bytes",
	"written_bytes",
	"read_bytes",
	"retry_count",
	"attempt_count",
	"attempt_ordinal",
	"dropped_count",
	"input_agent_message",
	"frame_count",
	"max_ms",
]);
const TOKEN_FIELDS = ["inputTokens", "cachedInputTokens", "outputTokens", "reasoningTokens", "totalTokens"];

function parsePositiveInteger(value, fallback) {
	const parsed = Number.parseInt(value ?? "", 10);
	return Number.isSafeInteger(parsed) && parsed > 0 ? parsed : fallback;
}

function parseArgs(argv) {
	const result = {
		directory: undefined,
		json: false,
		maxFiles: MAX_DEFAULT_FILES,
		maxBytes: MAX_DEFAULT_BYTES,
		maxLineBytes: MAX_DEFAULT_LINE_BYTES,
	};
	for (let index = 0; index < argv.length; index++) {
		const argument = argv[index];
		if (argument === "--dir") result.directory = argv[++index];
		else if (argument === "--json") result.json = true;
		else if (argument === "--max-files") result.maxFiles = parsePositiveInteger(argv[++index], result.maxFiles);
		else if (argument === "--max-bytes") result.maxBytes = parsePositiveInteger(argv[++index], result.maxBytes);
		else if (argument === "--max-line-bytes") {
			result.maxLineBytes = parsePositiveInteger(argv[++index], result.maxLineBytes);
		} else if (argument === "--help" || argument === "-h") {
			console.log(
				"Usage: summarize-performance-metrics.mjs --dir <metrics-root> [--json] " +
					"[--max-files N] [--max-bytes N] [--max-line-bytes N]",
			);
			process.exit(0);
		} else {
			throw new Error(`Unknown argument: ${argument}`);
		}
	}
	if (!result.directory) throw new Error("--dir is required");
	result.maxFiles = Math.min(result.maxFiles, 4096);
	result.maxBytes = Math.min(result.maxBytes, 1024 * 1024 * 1024);
	result.maxLineBytes = Math.min(result.maxLineBytes, 1024 * 1024);
	return result;
}

function newUsageSummary() {
	return {
		records: 0,
		fields: Object.fromEntries(TOKEN_FIELDS.map((field) => [field, { available: 0, sum: 0 }])),
		overlap: {
			cachedInputIncludedInInput: { true: 0, false: 0, unavailable: 0 },
			reasoningIncludedInOutput: { true: 0, false: 0, unavailable: 0 },
		},
	};
}

function newSummary() {
	return {
		schemaVersion: 1,
		files: {
			discovered: 0,
			read: 0,
			skippedByLimit: 0,
			readErrors: 0,
			bytesSelectedAtOpen: 0,
			bytesActuallyRead: 0,
			grewWhileReading: 0,
			truncatedByByteLimit: 0,
		},
		records: { valid: 0, invalid: 0, oversized: 0, truncated: 0, byOperation: {}, byOutcome: {}, startedByOperation: {} },
		sessions: { observed: 0, capped: false },
		measurements: { byOperation: {}, providerAttemptsByIdentity: {}, compactionAttemptsByIdentity: {}, identityGroupsCapped: false },
		usage: { provider: newUsageSummary(), localEstimate: newUsageSummary() },
		droppedRecordsReported: 0,
		notes: [
			"UI input spans begin when decoded terminal input reaches the host, not at physical keypress. UI acknowledgments end at the actual prompt RPC reply, not durable delivery or model completion; their local action IDs do not join daemon action IDs. UI menu spans include retrieval, mounting and the next completed render pass; session-open begins at terminal-host entry, not browser selection. UI render totals/counts aggregate requested render passes in approximately one-second windows; max_ms is the worst call, not a per-window latency. A render pass may produce no terminal bytes; these are not physical screen-paint timings.",
			"Duration distributions are separated by operation. Nested request/attempt spans and concurrent spans are never added as elapsed workflow time.",
			"Duration aggregates report count, unavailable, mean, and range; they intentionally omit a duration sum.",
			"Token fields are summed independently; overlap categories are never added into input/output or provider total.",
			"Provider usage and local estimates are separate. This report is not a bill or cost estimate.",
			"Current host integrations attach authoritative raw provider usage to provider_attempt only; schema-valid legacy records remain readable.",
			"Compaction start records (outcome \"started\", or a missing outcome in legacy files) are counted separately and excluded from completed measurements. Compaction provider attempts have their own identity distributions.",
			"File bytes selected from opening stats and bytes actually consumed are reported separately.",
		],
	};
}

function finiteNonnegative(value) {
	return typeof value === "number" && Number.isFinite(value) && value >= 0;
}

function newMeasurementAggregate(key) {
	return {
		available: 0,
		unavailable: 0,
		mean: null,
		min: null,
		max: null,
		...(key.endsWith("_ms") ? {} : { sum: 0 }),
	};
}

function addMeasurement(container, key, value) {
	const aggregate = (container[key] ??= newMeasurementAggregate(key));
	if (!finiteNonnegative(value)) {
		aggregate.unavailable++;
		return;
	}
	aggregate.available++;
	aggregate.mean = aggregate.mean === null ? value : aggregate.mean + (value - aggregate.mean) / aggregate.available;
	aggregate.min = aggregate.min === null ? value : Math.min(aggregate.min, value);
	aggregate.max = aggregate.max === null ? value : Math.max(aggregate.max, value);
	if ("sum" in aggregate) aggregate.sum += value;
}

function boundedIdentityPart(value) {
	return typeof value === "string" && value.length > 0 ? value.slice(0, 160) : "unavailable";
}

function providerIdentityKey(identity) {
	return `identity:${boundedIdentityPart(identity?.provider)}/${boundedIdentityPart(identity?.model)}/${boundedIdentityPart(identity?.api)}`;
}

function addUsage(summary, usage) {
	if (!usage || (usage.source !== "provider" && usage.source !== "local_estimate")) return;
	const target = usage.source === "provider" ? summary.usage.provider : summary.usage.localEstimate;
	target.records++;
	for (const field of TOKEN_FIELDS) {
		const value = usage[field];
		if (finiteNonnegative(value)) {
			target.fields[field].available++;
			target.fields[field].sum += value;
		}
	}
	for (const field of ["cachedInputIncludedInInput", "reasoningIncludedInOutput"]) {
		const value = usage[field];
		if (value === true) target.overlap[field].true++;
		else if (value === false) target.overlap[field].false++;
		else target.overlap[field].unavailable++;
	}
}

function acceptRecord(summary, sessions, record) {
	if (!record || record.schemaVersion !== 1 || !OPERATIONS.has(record.operation)) {
		summary.records.invalid++;
		return;
	}
	summary.records.valid++;
	summary.records.byOperation[record.operation] = (summary.records.byOperation[record.operation] ?? 0) + 1;
	if (typeof record.outcome === "string") {
		summary.records.byOutcome[record.outcome] = (summary.records.byOutcome[record.outcome] ?? 0) + 1;
	}
	const sessionId = record.correlation?.sessionId;
	if (typeof sessionId === "string" && sessions.size < SESSION_SET_LIMIT) sessions.add(sessionId);
	else if (typeof sessionId === "string") summary.sessions.capped = true;
	if (record.identity?.component === "compaction" && (record.outcome === undefined || record.outcome === "started")) {
		summary.records.startedByOperation[record.operation] = (summary.records.startedByOperation[record.operation] ?? 0) + 1;
		return;
	}
	if (record.measurements && typeof record.measurements === "object") {
		const operationMeasurements = (summary.measurements.byOperation[record.operation] ??= {});
		let identityMeasurements;
		if (record.operation === "provider_attempt") {
			let key = providerIdentityKey(record.identity);
			const identities = record.identity?.component === "compaction"
				? summary.measurements.compactionAttemptsByIdentity
				: summary.measurements.providerAttemptsByIdentity;
			if (!(key in identities) && Object.keys(identities).length >= IDENTITY_GROUP_LIMIT) {
				key = "identity:(other)";
				summary.measurements.identityGroupsCapped = true;
			}
			identityMeasurements = (identities[key] ??= {});
		}
		for (const [key, value] of Object.entries(record.measurements)) {
			if (!MEASUREMENTS.has(key)) continue;
			addMeasurement(operationMeasurements, key, value);
			if (identityMeasurements) addMeasurement(identityMeasurements, key, value);
		}
		const dropped = record.measurements.dropped_count;
		if (record.operation === "recorder" && finiteNonnegative(dropped)) summary.droppedRecordsReported += dropped;
	}
	addUsage(summary, record.usage);
}

function consumeLine(summary, sessions, line, oversized) {
	if (oversized) {
		summary.records.oversized++;
		return;
	}
	if (!line) return;
	try {
		acceptRecord(summary, sessions, JSON.parse(line));
	} catch {
		summary.records.invalid++;
	}
}

async function readBoundedLines(path, maxLineBytes, maxBytes, onLine) {
	if (maxBytes <= 0) return { bytesRead: 0, trailingLine: "", trailingOversized: false, error: false };
	const stream = createReadStream(path, {
		start: 0,
		end: maxBytes - 1,
		highWaterMark: Math.min(64 * 1024, maxBytes),
	});
	const decoder = new StringDecoder("utf8");
	let buffered = "";
	let bytesRead = 0;
	let discardingOversized = false;
	let readError = false;
	try {
		for await (const chunk of stream) {
			bytesRead += chunk.length;
			buffered += decoder.write(chunk);
			while (true) {
				const newline = buffered.indexOf("\n");
				if (newline < 0) break;
				const line = buffered.slice(0, newline);
				buffered = buffered.slice(newline + 1);
				if (discardingOversized || Buffer.byteLength(line, "utf8") > maxLineBytes) onLine(undefined, true);
				else if (line.length > 0) onLine(line, false);
				discardingOversized = false;
			}
			if (Buffer.byteLength(buffered, "utf8") > maxLineBytes) {
				buffered = "";
				discardingOversized = true;
			}
		}
		buffered += decoder.end();
	} catch {
		readError = true;
	}
	return {
		bytesRead,
		trailingLine: discardingOversized ? "" : buffered,
		trailingOversized: discardingOversized || Buffer.byteLength(buffered, "utf8") > maxLineBytes,
		error: readError,
	};
}

function printText(summary) {
	console.log(
		`records: ${summary.records.valid} valid, ${summary.records.invalid} invalid, ` +
			`${summary.records.oversized} oversized, ${summary.records.truncated} truncated`,
	);
	console.log(
		`files: ${summary.files.read}/${summary.files.discovered} read; ` +
			`${summary.files.bytesActuallyRead}/${summary.files.bytesSelectedAtOpen} actual/selected bytes`,
	);
	console.log(`sessions observed: ${summary.sessions.observed}${summary.sessions.capped ? "+ (capped)" : ""}`);
	for (const [operation, measurements] of Object.entries(summary.measurements.byOperation).sort()) {
		for (const [name, aggregate] of Object.entries(measurements).sort()) {
			const sum = "sum" in aggregate ? `, sum=${aggregate.sum}` : "";
			console.log(
				`${operation}.${name}: n=${aggregate.available}, unavailable=${aggregate.unavailable}, ` +
					`mean=${aggregate.mean ?? "n/a"}, range=${aggregate.min ?? "n/a"}..${aggregate.max ?? "n/a"}${sum}`,
			);
		}
	}
	for (const [source, usage] of Object.entries(summary.usage)) {
		const fields = TOKEN_FIELDS.map(
			(field) => `${field}=${usage.fields[field].sum} (n=${usage.fields[field].available})`,
		).join(", ");
		console.log(`usage ${source}: ${fields}`);
	}
	console.log(`dropped records reported: ${summary.droppedRecordsReported}`);
	console.log(summary.notes.join(" "));
}

async function main() {
	const options = parseArgs(process.argv.slice(2));
	const directory = resolve(options.directory);
	const entries = (await readdir(directory, { withFileTypes: true }))
		.filter((entry) => entry.isFile() && /^performance-v1-.*\.jsonl(?:\.\d+)?$/.test(entry.name))
		.map((entry) => entry.name)
		.sort();
	const summary = newSummary();
	summary.files.discovered = entries.length;
	const sessions = new Set();
	let reachedByteLimit = false;
	for (const name of entries.slice(0, options.maxFiles)) {
		if (reachedByteLimit) {
			summary.files.skippedByLimit++;
			continue;
		}
		const path = resolve(directory, name);
		try {
			const sizeAtOpen = (await stat(path)).size;
			const remainingBytes = options.maxBytes - summary.files.bytesActuallyRead;
			if (remainingBytes <= 0) {
				reachedByteLimit = true;
				summary.files.skippedByLimit++;
				continue;
			}
			summary.files.bytesSelectedAtOpen += sizeAtOpen;
			summary.files.read++;
			const result = await readBoundedLines(path, options.maxLineBytes, remainingBytes, (line, oversized) => {
				consumeLine(summary, sessions, line, oversized);
			});
			summary.files.bytesActuallyRead += result.bytesRead;
			if (result.error) summary.files.readErrors++;
			let sizeAtClose = sizeAtOpen;
			try {
				sizeAtClose = (await stat(path)).size;
			} catch {
				summary.files.readErrors++;
			}
			if (sizeAtClose > sizeAtOpen) summary.files.grewWhileReading++;
			const truncated = sizeAtClose > result.bytesRead;
			if (truncated) {
				summary.files.truncatedByByteLimit++;
				summary.records.truncated++;
			} else {
				consumeLine(summary, sessions, result.trailingLine, result.trailingOversized);
			}
			if (summary.files.bytesActuallyRead >= options.maxBytes) reachedByteLimit = true;
		} catch {
			summary.files.readErrors++;
		}
	}
	summary.files.skippedByLimit += Math.max(0, entries.length - options.maxFiles);
	summary.sessions.observed = sessions.size;
	if (options.json) console.log(JSON.stringify(summary, null, 2));
	else printText(summary);
}

main().catch((error) => {
	console.error(error instanceof Error ? error.message : String(error));
	process.exitCode = 1;
});
