import { randomUUID } from "node:crypto";
import { appendFile, mkdir, rename, rm, stat } from "node:fs/promises";
import { join, resolve } from "node:path";
import {
	PERFORMANCE_METRICS_SCHEMA_VERSION,
	type PerformanceMetricComponent,
	type PerformanceMetricCorrelation,
	type PerformanceMetricEvent,
	type PerformanceMetricIdentity,
	type PerformanceMetricMeasurement,
	type PerformanceMetricOperation,
	type PerformanceMetricOutcome,
	type PerformanceMetricRecorder,
	type PerformanceMetricRecordV1,
	type PerformanceMetricUsageV1,
} from "@earendil-works/pi-agent-core";

const DEFAULT_MAX_BUFFERED_RECORDS = 512;
const DEFAULT_MAX_BUFFERED_BYTES = 256 * 1024;
const DEFAULT_MAX_RECORD_BYTES = 8 * 1024;
const DEFAULT_MAX_FILE_BYTES = 4 * 1024 * 1024;
const DEFAULT_MAX_FILES = 4;
const DEFAULT_FLUSH_INTERVAL_MS = 1_000;
const DEFAULT_CLOSE_TIMEOUT_MS = 1_000;

const OPERATIONS = new Set<PerformanceMetricOperation>([
	"logical_request",
	"provider_attempt",
	"tool",
	"snapshot",
	"compaction",
	"file_retry",
	"session_reopen",
	"ui_input",
	"ui_input_ack",
	"ui_render",
	"ui_menu_open",
	"ui_session_open",
	"recorder",
]);
const OUTCOMES = new Set<PerformanceMetricOutcome>(["started", "success", "failure", "cancelled", "unavailable"]);
const COMPONENTS = new Set<PerformanceMetricComponent>([
	"agent",
	"provider",
	"tool",
	"snapshot",
	"compaction",
	"persistence",
	"session",
	"recorder",
]);
const MEASUREMENTS = new Set<PerformanceMetricMeasurement>([
	"total_ms",
	"wait_ms",
	"dispatch_to_response_headers_ms",
	"dispatch_to_first_event_ms",
	"dispatch_to_first_visible_ms",
	"local_gateway_wait_ms",
	"upstream_wait_ms",
	"serialization_ms",
	"serialization_cpu_ms",
	"serialization_max_variable_ms",
	"serialization_slow_variables",
	"serialization_saved_ms",
	"serialization_skipped_ms",
	"frame_count",
	"max_ms",
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
]);

export interface PerformanceMetricFileIO {
	mkdir(path: string): Promise<void>;
	append(path: string, data: string): Promise<void>;
	size(path: string): Promise<number | null>;
	rename(source: string, destination: string): Promise<void>;
	remove(path: string): Promise<void>;
}

export interface LocalPerformanceMetricRecorderOptions {
	directory: string;
	sessionId: string;
	maxBufferedRecords?: number;
	maxBufferedBytes?: number;
	maxRecordBytes?: number;
	maxFileBytes?: number;
	maxFiles?: number;
	flushIntervalMs?: number;
	/** Best-effort shutdown wait. Telemetry never owns process/session liveness. */
	closeTimeoutMs?: number;
	monotonicNow?: () => number;
	wallNow?: () => number;
	randomId?: () => string;
	fileIO?: PerformanceMetricFileIO;
}

export interface EnvironmentPerformanceMetricRecorderOptions
	extends Omit<LocalPerformanceMetricRecorderOptions, "directory"> {
	agentDir: string;
	env?: Readonly<Record<string, string | undefined>>;
}

const defaultFileIO: PerformanceMetricFileIO = {
	async mkdir(path) {
		await mkdir(path, { recursive: true });
	},
	async append(path, data) {
		await appendFile(path, data, { encoding: "utf8", mode: 0o600 });
	},
	async size(path) {
		try {
			return (await stat(path)).size;
		} catch (error) {
			if (isErrorCode(error, "ENOENT")) return null;
			throw error;
		}
	},
	async rename(source, destination) {
		await rename(source, destination);
	},
	async remove(path) {
		await rm(path, { force: true });
	},
};

function isErrorCode(error: unknown, code: string): boolean {
	return typeof error === "object" && error !== null && "code" in error && error.code === code;
}

function boundedInteger(value: number | undefined, fallback: number, minimum: number, maximum: number): number {
	if (value === undefined || !Number.isFinite(value)) return fallback;
	return Math.min(maximum, Math.max(minimum, Math.floor(value)));
}

function sanitizeString(value: unknown, maxLength: number): string | undefined {
	if (typeof value !== "string") return undefined;
	const sanitized = value.replace(/[\u0000-\u001f\u007f]/g, "?").slice(0, maxLength);
	return sanitized.length > 0 ? sanitized : undefined;
}

function sanitizeNullableString(value: unknown, maxLength: number): string | null | undefined {
	if (value === null) return null;
	return sanitizeString(value, maxLength);
}

function sanitizeMeasurement(value: unknown): number | null {
	if (value === null) return null;
	if (typeof value !== "number" || !Number.isFinite(value) || value < 0) return null;
	return Math.min(value, Number.MAX_SAFE_INTEGER);
}

function sanitizeTokenCount(value: unknown): number | null {
	const sanitized = sanitizeMeasurement(value);
	return sanitized === null ? null : Math.floor(sanitized);
}

function sanitizeCorrelation(value: unknown): PerformanceMetricCorrelation {
	if (typeof value !== "object" || value === null) return {};
	const correlation = value as Record<string, unknown>;
	return {
		logicalRequestId: sanitizeString(correlation.logicalRequestId, 128),
		providerAttemptId: sanitizeString(correlation.providerAttemptId, 128),
		toolCallId: sanitizeString(correlation.toolCallId, 128),
	};
}

function sanitizeIdentity(value: unknown): PerformanceMetricIdentity | undefined {
	if (typeof value !== "object" || value === null) return undefined;
	const identity = value as Record<string, unknown>;
	const component = COMPONENTS.has(identity.component as PerformanceMetricComponent)
		? (identity.component as PerformanceMetricComponent)
		: undefined;
	const sanitized: PerformanceMetricIdentity = {
		provider: sanitizeNullableString(identity.provider, 96),
		model: sanitizeNullableString(identity.model, 160),
		api: sanitizeNullableString(identity.api, 96),
		component,
	};
	return Object.values(sanitized).some((item) => item !== undefined) ? sanitized : undefined;
}

function sanitizeMeasurements(
	value: unknown,
): Partial<Record<PerformanceMetricMeasurement, number | null>> | undefined {
	if (typeof value !== "object" || value === null) return undefined;
	const source = value as Record<string, unknown>;
	const sanitized: Partial<Record<PerformanceMetricMeasurement, number | null>> = {};
	for (const [key, measurement] of Object.entries(source)) {
		if (MEASUREMENTS.has(key as PerformanceMetricMeasurement)) {
			sanitized[key as PerformanceMetricMeasurement] = sanitizeMeasurement(measurement);
		}
	}
	return Object.keys(sanitized).length > 0 ? sanitized : undefined;
}

function sanitizeUsage(value: unknown): PerformanceMetricUsageV1 | undefined {
	if (typeof value !== "object" || value === null) return undefined;
	const source = value as Record<string, unknown>;
	if (source.source !== "provider" && source.source !== "local_estimate") return undefined;
	const boolOrNull = (item: unknown): boolean | null => (typeof item === "boolean" ? item : null);
	const usage: PerformanceMetricUsageV1 = {
		source: source.source,
		inputTokens: sanitizeTokenCount(source.inputTokens),
		cachedInputTokens: sanitizeTokenCount(source.cachedInputTokens),
		outputTokens: sanitizeTokenCount(source.outputTokens),
		reasoningTokens: sanitizeTokenCount(source.reasoningTokens),
		totalTokens: sanitizeTokenCount(source.totalTokens),
		cachedInputIncludedInInput: boolOrNull(source.cachedInputIncludedInInput),
		reasoningIncludedInOutput: boolOrNull(source.reasoningIncludedInOutput),
	};
	if (source.source === "local_estimate") {
		usage.estimator = sanitizeString(source.estimator, 64) ?? "unspecified";
	}
	return usage;
}

function optInEnabled(value: string | undefined): boolean {
	return value !== undefined && ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
}

function safeFileSegment(value: string, fallback: string): string {
	const sanitized = value
		.replace(/[^A-Za-z0-9_-]/g, "-")
		.replace(/-+/g, "-")
		.slice(0, 48);
	return sanitized || fallback;
}

function settleWithin(promise: Promise<unknown>, timeoutMs: number): Promise<void> {
	return new Promise((resolvePromise) => {
		let settled = false;
		const finish = () => {
			if (settled) return;
			settled = true;
			clearTimeout(timer);
			resolvePromise();
		};
		const timer = setTimeout(finish, timeoutMs);
		timer.unref?.();
		void promise.then(finish, finish);
	});
}

export class LocalPerformanceMetricRecorder implements PerformanceMetricRecorder {
	readonly sessionId: string;
	readonly logPath: string;
	private readonly directory: string;
	private readonly maxBufferedRecords: number;
	private readonly maxBufferedBytes: number;
	private readonly maxRecordBytes: number;
	private readonly maxFileBytes: number;
	private readonly maxFiles: number;
	private readonly closeTimeoutMs: number;
	private readonly now: () => number;
	private readonly wallNow: () => number;
	private readonly randomId: () => string;
	private readonly fileIO: PerformanceMetricFileIO;
	private readonly timer: ReturnType<typeof setInterval>;
	private bufferedLines: string[] = [];
	private bufferedBytes = 0;
	private pendingDroppedRecords = 0;
	private sequence = 0;
	private closed = false;
	private flushRequested = false;
	private flushInFlight: Promise<void> | undefined;
	private closePromise: Promise<void> | undefined;

	constructor(options: LocalPerformanceMetricRecorderOptions) {
		this.directory = resolve(options.directory);
		this.sessionId = sanitizeString(options.sessionId, 128) ?? "unavailable";
		this.maxFileBytes = boundedInteger(options.maxFileBytes, DEFAULT_MAX_FILE_BYTES, 4 * 1024, 64 * 1024 * 1024);
		this.maxRecordBytes = Math.min(
			Math.floor(this.maxFileBytes / 2),
			boundedInteger(options.maxRecordBytes, DEFAULT_MAX_RECORD_BYTES, 512, 16 * 1024),
		);
		this.maxBufferedBytes = Math.min(
			this.maxFileBytes - this.maxRecordBytes,
			boundedInteger(options.maxBufferedBytes, DEFAULT_MAX_BUFFERED_BYTES, 1024, 4 * 1024 * 1024),
		);
		this.maxBufferedRecords = boundedInteger(options.maxBufferedRecords, DEFAULT_MAX_BUFFERED_RECORDS, 1, 4096);
		this.maxFiles = boundedInteger(options.maxFiles, DEFAULT_MAX_FILES, 1, 16);
		this.closeTimeoutMs = boundedInteger(options.closeTimeoutMs, DEFAULT_CLOSE_TIMEOUT_MS, 1, 10_000);
		this.now = options.monotonicNow ?? (() => globalThis.performance.now());
		this.wallNow = options.wallNow ?? Date.now;
		this.randomId = options.randomId ?? randomUUID;
		this.fileIO = options.fileIO ?? defaultFileIO;
		const instanceId = safeFileSegment(this.randomId(), "instance");
		const sessionSegment = safeFileSegment(this.sessionId, "session");
		this.logPath = join(this.directory, `performance-v1-${sessionSegment}-${instanceId}.jsonl`);
		const flushIntervalMs = boundedInteger(options.flushIntervalMs, DEFAULT_FLUSH_INTERVAL_MS, 100, 60_000);
		this.timer = setInterval(() => {
			void this.flush();
		}, flushIntervalMs);
		this.timer.unref?.();
	}

	monotonicNow(): number {
		return this.now();
	}

	nextId(scope: "logical_request" | "provider_attempt"): string {
		return `${scope}-${this.randomId()}`;
	}

	record(event: PerformanceMetricEvent): void {
		if (this.closed) return;
		try {
			const line = this.serializeEvent(event);
			if (line === undefined) {
				this.noteDroppedRecord();
				return;
			}
			const bytes = Buffer.byteLength(line, "utf8");
			if (
				bytes > this.maxRecordBytes ||
				this.bufferedLines.length >= this.maxBufferedRecords ||
				this.bufferedBytes + bytes > this.maxBufferedBytes
			) {
				this.noteDroppedRecord();
				return;
			}
			this.bufferedLines.push(line);
			this.bufferedBytes += bytes;
		} catch {
			this.noteDroppedRecord();
		}
	}

	flush(): Promise<void> {
		// Coalesce all callers onto one flight. At most one follow-up drain is
		// represented by a boolean, so stalled file I/O cannot grow a promise queue.
		if (this.flushInFlight) {
			this.flushRequested = true;
			return this.flushInFlight;
		}

		this.flushRequested = true;
		let flight: Promise<void>;
		flight = Promise.resolve().then(async () => {
			try {
				do {
					this.flushRequested = false;
					await this.flushOnce();
				} while (this.flushRequested);
			} catch {
				// Telemetry is disposable. flushOnce normally accounts for lost
				// records, but no unexpected sink error may escape to agent work.
			} finally {
				if (this.flushInFlight === flight) this.flushInFlight = undefined;
			}
		});
		this.flushInFlight = flight;
		return flight;
	}

	close(): Promise<void> {
		if (this.closePromise) return this.closePromise;
		this.closed = true;
		clearInterval(this.timer);
		// A stuck filesystem must not block session disposal. The write may still
		// finish in the background, but close resolves after this strict bound.
		this.closePromise = settleWithin(this.flush(), this.closeTimeoutMs);
		return this.closePromise;
	}

	private noteDroppedRecord(count = 1): void {
		this.pendingDroppedRecords = Math.min(Number.MAX_SAFE_INTEGER, this.pendingDroppedRecords + count);
	}

	private serializeEvent(event: PerformanceMetricEvent): string | undefined {
		if (!OPERATIONS.has(event.operation)) return undefined;
		const recordedAtMs = this.wallNow();
		if (!Number.isFinite(recordedAtMs)) return undefined;
		const correlation = sanitizeCorrelation(event.correlation);
		const record: PerformanceMetricRecordV1 = {
			schemaVersion: PERFORMANCE_METRICS_SCHEMA_VERSION,
			sequence: ++this.sequence,
			recordedAt: new Date(recordedAtMs).toISOString(),
			operation: event.operation,
			correlation: { sessionId: this.sessionId, ...correlation },
		};
		const identity = sanitizeIdentity(event.identity);
		if (identity) record.identity = identity;
		if (event.outcome && OUTCOMES.has(event.outcome)) record.outcome = event.outcome;
		const measurements = sanitizeMeasurements(event.measurements);
		if (measurements) record.measurements = measurements;
		const usage = sanitizeUsage(event.usage);
		if (usage) record.usage = usage;
		return `${JSON.stringify(record)}\n`;
	}

	private async flushOnce(): Promise<void> {
		if (this.bufferedLines.length === 0 && this.pendingDroppedRecords === 0) return;
		const lines = this.bufferedLines;
		const bufferedRecordCount = lines.length;
		const previouslyDropped = this.pendingDroppedRecords;
		this.bufferedLines = [];
		this.bufferedBytes = 0;
		this.pendingDroppedRecords = 0;

		try {
			if (previouslyDropped > 0) {
				const dropLine = this.serializeEvent({
					operation: "recorder",
					identity: { component: "recorder" },
					outcome: "unavailable",
					measurements: { dropped_count: previouslyDropped },
				});
				if (!dropLine) throw new Error("Unable to encode recorder drop metric");
				lines.unshift(dropLine);
			}
			await this.fileIO.mkdir(this.directory);
			await this.writeLines(lines);
		} catch {
			this.noteDroppedRecord(previouslyDropped + bufferedRecordCount);
		}
	}

	private async writeLines(lines: string[]): Promise<void> {
		let chunk: string[] = [];
		let chunkBytes = 0;
		for (const line of lines) {
			const bytes = Buffer.byteLength(line, "utf8");
			if (bytes > this.maxFileBytes) {
				this.noteDroppedRecord();
				continue;
			}
			if (chunkBytes > 0 && chunkBytes + bytes > this.maxFileBytes) {
				await this.appendChunk(chunk.join(""), chunkBytes);
				chunk = [];
				chunkBytes = 0;
			}
			chunk.push(line);
			chunkBytes += bytes;
		}
		if (chunkBytes > 0) await this.appendChunk(chunk.join(""), chunkBytes);
	}

	private async appendChunk(data: string, bytes: number): Promise<void> {
		const currentSize = (await this.fileIO.size(this.logPath)) ?? 0;
		if (currentSize + bytes > this.maxFileBytes) await this.rotate();
		await this.fileIO.append(this.logPath, data);
	}

	private async rotate(): Promise<void> {
		if (this.maxFiles === 1) {
			await this.fileIO.remove(this.logPath);
			return;
		}
		for (let index = this.maxFiles - 1; index >= 1; index--) {
			const source = index === 1 ? this.logPath : `${this.logPath}.${index - 1}`;
			const destination = `${this.logPath}.${index}`;
			await this.fileIO.remove(destination);
			try {
				await this.fileIO.rename(source, destination);
			} catch (error) {
				if (!isErrorCode(error, "ENOENT")) throw error;
			}
		}
	}
}

export function createLocalPerformanceMetricRecorder(
	options: LocalPerformanceMetricRecorderOptions,
): LocalPerformanceMetricRecorder | undefined {
	try {
		return new LocalPerformanceMetricRecorder(options);
	} catch {
		return undefined;
	}
}

export function createLocalPerformanceMetricRecorderFromEnvironment(
	options: EnvironmentPerformanceMetricRecorderOptions,
): LocalPerformanceMetricRecorder | undefined {
	const env = options.env ?? process.env;
	if (!optInEnabled(env.PRIME_AGENT_PERFORMANCE_METRICS)) return undefined;
	const configuredDirectory = sanitizeString(env.PRIME_AGENT_PERFORMANCE_METRICS_DIR, 2048);
	const { agentDir, ...recorderOptions } = options;
	delete recorderOptions.env;
	return createLocalPerformanceMetricRecorder({
		...recorderOptions,
		directory: configuredDirectory ?? join(agentDir, "performance-metrics"),
	});
}
