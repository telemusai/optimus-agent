import type { AssistantMessage } from "@earendil-works/pi-ai";

export const PERFORMANCE_METRICS_SCHEMA_VERSION = 1 as const;

export type PerformanceMetricOperation =
	| "logical_request"
	| "provider_attempt"
	| "tool"
	| "snapshot"
	| "compaction"
	| "file_retry"
	| "session_reopen"
	| "ui_input"
	| "ui_input_ack"
	| "ui_render"
	| "ui_menu_open"
	| "ui_session_open"
	| "recorder";

export type PerformanceMetricOutcome = "started" | "success" | "failure" | "cancelled" | "unavailable";

export type PerformanceMetricMeasurement =
	| "total_ms"
	| "wait_ms"
	| "dispatch_to_response_headers_ms"
	| "dispatch_to_first_event_ms"
	| "dispatch_to_first_visible_ms"
	| "local_gateway_wait_ms"
	| "upstream_wait_ms"
	| "serialization_ms"
	| "serialization_cpu_ms"
	| "serialization_max_variable_ms"
	| "serialization_slow_variables"
	| "serialization_saved_ms"
	| "serialization_skipped_ms"
	| "frame_count"
	| "max_ms"
	| "write_ms"
	| "queue_ms"
	| "next_cell_delay_ms"
	| "reopen_ms"
	| "serialized_bytes"
	| "written_bytes"
	| "read_bytes"
	| "retry_count"
	| "attempt_count"
	| "attempt_ordinal"
	| "dropped_count";

export type PerformanceMetricComponent =
	| "agent"
	| "provider"
	| "tool"
	| "snapshot"
	| "compaction"
	| "persistence"
	| "session"
	| "recorder";

export interface PerformanceMetricCorrelation {
	logicalRequestId?: string;
	providerAttemptId?: string;
	toolCallId?: string;
}

export interface PerformanceMetricIdentity {
	provider?: string | null;
	model?: string | null;
	api?: string | null;
	component?: PerformanceMetricComponent;
}

/**
 * Provider totals and categories must not be summed again. Overlap fields stay
 * null unless the emitting provider establishes their exact semantics.
 */
export interface PerformanceMetricUsageV1 {
	source: "provider" | "local_estimate";
	inputTokens: number | null;
	cachedInputTokens: number | null;
	outputTokens: number | null;
	reasoningTokens: number | null;
	totalTokens: number | null;
	cachedInputIncludedInInput: boolean | null;
	reasoningIncludedInOutput: boolean | null;
	estimator?: string;
}

export interface PerformanceMetricEvent {
	operation: PerformanceMetricOperation;
	correlation?: PerformanceMetricCorrelation;
	identity?: PerformanceMetricIdentity;
	outcome?: PerformanceMetricOutcome;
	measurements?: Partial<Record<PerformanceMetricMeasurement, number | null>>;
	usage?: PerformanceMetricUsageV1;
}

export interface PerformanceMetricRecordV1 extends Omit<PerformanceMetricEvent, "correlation"> {
	schemaVersion: typeof PERFORMANCE_METRICS_SCHEMA_VERSION;
	sequence: number;
	recordedAt: string;
	correlation: PerformanceMetricCorrelation & { sessionId: string };
}

/** A recorder must never make agent work wait for telemetry persistence. */
export interface PerformanceMetricRecorder {
	readonly sessionId: string;
	monotonicNow(): number;
	nextId(scope: "logical_request" | "provider_attempt"): string;
	record(event: PerformanceMetricEvent): void;
	flush(): Promise<void>;
	close(): Promise<void>;
}

/** @internal Process-local exactly-once state shared by attempts in one host retry group. */
export interface AgentLoopLogicalRequestSettlement {
	settled: boolean;
	maxProviderAttemptNumber: number;
}

/** Optional correlation supplied by a host around one low-level agent run. */
export interface AgentLoopPerformanceMetrics {
	recorder: PerformanceMetricRecorder;
	logicalRequestId?: string;
	logicalRequestStartedAt?: number;
	/** Host-observed stream invocation ordinal, not an SDK-internal retry count. */
	providerAttemptNumber?: number;
	/**
	 * Defers the outer logical-request terminal to a host that groups local retries.
	 * The Agent loop still records each locally observed provider attempt.
	 */
	hostOwnsLogicalRequestTerminal?: boolean;
	/** @internal Shared only across the host-observed attempts of this logical request. */
	logicalRequestSettlement?: AgentLoopLogicalRequestSettlement;
}

export function elapsedMetricMs(start: number | undefined, end: number | undefined): number | null {
	if (start === undefined || end === undefined || !Number.isFinite(start) || !Number.isFinite(end) || end < start) {
		return null;
	}
	return end - start;
}

/** Defensively contains third-party recorder failures at every call site. */
export function safeRecordPerformanceMetric(
	recorder: PerformanceMetricRecorder | undefined,
	event: PerformanceMetricEvent,
): void {
	if (!recorder) return;
	try {
		recorder.record(event);
	} catch {
		// Performance telemetry is disposable and must not change agent behavior.
	}
}

function normalizedPositiveTokenCount(value: number): number | null {
	// Normalized Usage uses zero both for an explicit zero and for a missing raw
	// field. Without the provider observation hook, only positive values prove
	// field-level availability.
	return Number.isFinite(value) && value > 0 ? value : null;
}

/**
 * Converts existing normalized usage without treating its all-zero placeholder
 * as proof that a provider reported usage.
 */
export function performanceMetricUsageFromAssistant(message: AssistantMessage): PerformanceMetricUsageV1 {
	const usage = message.usage;
	const hasAuthoritativeUsage = [usage.input, usage.cacheRead, usage.output, usage.totalTokens].some(
		(value) => Number.isFinite(value) && value > 0,
	);
	if (!hasAuthoritativeUsage) {
		return {
			source: "provider",
			inputTokens: null,
			cachedInputTokens: null,
			outputTokens: null,
			reasoningTokens: null,
			totalTokens: null,
			cachedInputIncludedInInput: null,
			reasoningIncludedInOutput: null,
		};
	}

	return {
		source: "provider",
		inputTokens: normalizedPositiveTokenCount(usage.input),
		cachedInputTokens: normalizedPositiveTokenCount(usage.cacheRead),
		outputTokens: normalizedPositiveTokenCount(usage.output),
		reasoningTokens: null,
		totalTokens: normalizedPositiveTokenCount(usage.totalTokens),
		// Future/custom provider normalizers may use a different overlap contract.
		cachedInputIncludedInInput: null,
		// The normalized Usage type does not currently retain this provider detail.
		reasoningIncludedInOutput: null,
	};
}
