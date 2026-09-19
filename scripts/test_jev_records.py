"""Offline regression tests for metadata-only Jev reports."""

import json
import tempfile
import unittest
from pathlib import Path

from summarize_jev_records import read_records, summarize


class JevReportTests(unittest.TestCase):
    def row(self, schema="jev.compare/1", **changes):
        row = {
            "schema_version": schema, "session_id": "s1", "request_id": "r1",
            "question_id": "complexity.0", "mode": "compare-active",
            "compaction_enabled": False, "attempt": 1, "duration_ms": 30,
            "agreement": "noncomparable", "applied": False,
        }
        row.update(changes)
        return row

    def test_combined_rows_share_one_request_and_question(self):
        rows = [self.row(), self.row("jev.active/1", applied=True, acceptance="accepted")]
        group = summarize(rows)["groups"][0]
        self.assertEqual(group["jev_requests"], 1)
        self.assertEqual(group["jev_unique_questions"], 1)
        self.assertEqual(group["jev_latency_ms"]["samples"], 1)
        self.assertEqual(group["counts"]["applied"], 1)

    def test_attempts_count_once_for_multiple_questions_and_duplicate_rows(self):
        rows = [self.row(attempt=3), self.row(question_id="tools.0", attempt=3), self.row(attempt=3)]
        group = summarize(rows)["groups"][0]
        self.assertEqual(group["jev_transport_attempts"], 3)
        self.assertEqual(group["jev_unique_questions"], 2)
        self.assertEqual(group["counts"]["comparison_questions"], 2)

    def test_off_compaction_on_is_independent(self):
        row = self.row("jev.compaction/1", mode="off", compaction_enabled=True,
                       observed_metrics={"messages_before": 20, "messages_after": 10,
                                         "chars_before": 1000, "chars_after": 400,
                                         "calls_removed": 5, "question_count": 12})
        group = summarize([row])["groups"][0]
        self.assertEqual((group["mode"], group["compaction"]), ("off", "on"))
        self.assertEqual(group["compaction_totals"]["calls_removed"], 5)
        self.assertEqual(group["compaction_totals"]["reduction_ratio"], 0.6)
        self.assertEqual(group["jev_unique_questions"], 12)
        self.assertIsNone(group["compaction_totals"]["results_truncated"])

    def test_jev_usage_is_deduplicated_and_incomplete_totals_stay_unknown(self):
        metrics = {"jev_input_tokens": 123, "jev_output_tokens": 17}
        rows = [self.row(observed_metrics=metrics), self.row("jev.active/1", observed_metrics=metrics)]
        group = summarize(rows)["groups"][0]
        self.assertEqual(group["jev_usage_tokens"]["jev_input_tokens"], 123)
        self.assertEqual(group["jev_usage_tokens"]["jev_output_tokens"], 17)
        rows.append(self.row(request_id="unknown-usage"))
        self.assertIsNone(summarize(rows)["groups"][0]["jev_usage_tokens"]["jev_input_tokens"])

    def test_unmeasured_and_partial_totals_are_unknown(self):
        first = self.row("jev.run/1", primary_llm_calls=3)
        second = self.row("jev.run/1", request_id="r2")
        group = summarize([first, second])["groups"][0]
        self.assertIsNone(group["run_totals"]["primary_llm_calls"])
        self.assertIsNone(group["run_totals"]["elapsed_ms"])

    def test_run_metadata_is_not_a_jev_network_call(self):
        group = summarize([self.row("jev.run/1", observed_metrics={"primary_llm_calls": 3})])["groups"][0]
        self.assertEqual(group["jev_requests"], 0)
        self.assertEqual(group["jev_unique_questions"], 0)
        self.assertEqual(group["run_totals"]["primary_llm_calls"], 3)

    def test_compaction_batches_count_questions_not_history_twice(self):
        rows = [self.row("jev.compaction/1", observed_metrics={"question_count": 4}),
                self.row("jev.compaction/1", request_id="r2",
                         observed_metrics={"question_count": 6, "chars_before": 1000, "chars_after": 400})]
        group = summarize(rows)["groups"][0]
        self.assertEqual(group["jev_requests"], 2)
        self.assertEqual(group["jev_unique_questions"], 10)
        self.assertEqual(group["compaction_totals"]["samples"], 1)

    def test_skip_before_dispatch_is_not_a_network_request(self):
        group = summarize([self.row(attempt=0, skipped_reason="mode_disabled")])["groups"][0]
        self.assertEqual(group["jev_requests"], 0)
        self.assertEqual(group["jev_unique_questions"], 0)
        self.assertIsNone(group["jev_latency_ms"]["mean"])

    def test_interrupted_attempt_count_is_a_lower_bound_not_exact(self):
        group = summarize([self.row("jev.active/1", attempt=1, attempt_count_known=False)])["groups"][0]
        self.assertEqual(group["jev_requests"], 1)
        self.assertIsNone(group["jev_transport_attempts"])
        self.assertEqual(group["jev_transport_attempts_lower_bound"], 1)

    def test_retrieval_metadata_and_confidence_deduplicate_combined_questions(self):
        baseline = {"candidate_count": "2", "retained_count": "2", "removed_count": "0",
                    "estimated_candidate_tokens_before": "100", "estimated_candidate_tokens_after": "100"}
        actual = {**baseline, "retained_count": "1", "removed_count": "1", "estimated_candidate_tokens_after": "40"}
        rows = [self.row(category="memory_relevance", confidence=0.99, baseline_action=baseline),
                self.row("jev.active/1", category="memory_relevance", confidence=0.99,
                         baseline_action=baseline, actual_action=actual)]
        group = summarize(rows)["groups"][0]
        self.assertEqual(group["retrieval_totals"]["memory_relevance"]["samples"], 1)
        self.assertEqual(group["retrieval_totals"]["memory_relevance"]["removed_count"], 1)
        self.assertEqual(group["retrieval_totals"]["memory_relevance"]["estimated_candidate_tokens_after"], 40)
        self.assertEqual(group["raw_confidence_distribution"]["0.9_to_1"], 1)

    def test_refused_recommendation_is_not_applied(self):
        group = summarize([self.row("jev.active/1", acceptance="fallback", outcome="refused",
                          selected_value="high", fallback_reason="low_confidence")])["groups"][0]
        self.assertEqual(group["counts"]["recommendations"], 1)
        self.assertEqual(group["counts"]["refused"], 1)
        self.assertEqual(group["counts"]["applied"], 0)

    def test_no_raw_values_or_unknown_labels_are_printed(self):
        secret = "not-a-real-secret-password"
        report = summarize([self.row(selected_value=secret, mode=secret, outcome=secret), {"body": secret}])
        self.assertNotIn(secret, json.dumps(report))
        self.assertEqual(report["ignored_records"], 1)
        self.assertEqual(report["groups"][0]["mode"], "unknown")

    def test_invalid_numeric_metadata_cannot_pollute_json(self):
        group = summarize([self.row(attempt=True, duration_ms=float("nan"))])["groups"][0]
        self.assertEqual(group["jev_requests"], 0)
        json.dumps(group, allow_nan=False)

    def test_wrong_shape_metadata_does_not_crash_or_echo_values(self):
        rows = [self.row(schema_version=[]), self.row(mode={}, category=[], agreement={}),
                self.row("jev.active/1", acceptance={}, category={})]
        report = summarize(rows)
        self.assertEqual(report["ignored_records"], 1)
        json.dumps(report, allow_nan=False)

    def test_reader_preserves_good_rows_and_counts_malformed_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            path.write_bytes(json.dumps(self.row()).encode() + b"\nnot-json\n")
            rows, status = read_records(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(status["malformed_lines"], 1)
        self.assertFalse(status["input_truncated"])


if __name__ == "__main__":
    unittest.main()
