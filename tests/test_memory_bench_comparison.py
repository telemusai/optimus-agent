"""Synthetic, offline evidence comparison tests; no scorer or dataset imports."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("compare_memory_bench", ROOT / "scripts/compare_memory_bench.py")
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


def record(qid, **updates):
    value = {"qid": qid, "answer_text": "synthetic answer", "variant": "base",
             "benchmark": "locomo_plus", "run_id": "synthetic-run",
             "env_id": "synthetic-env", "question": "Synthetic question?",
             "eval": {"type": "llm_judge"}}
    value.update(updates)
    return value


def score(qid, value=1.0, **updates):
    result = {"qid": qid, "score": value, "flag": None,
              "eval_type": "llm_judge", "env_id": "synthetic-env"}
    result.update(updates)
    return result


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory(prefix="memory-comparison-")
        self.addCleanup(scratch.cleanup)
        self.root = Path(scratch.name)

    def arm(self, name, records=None, scores=None, **artifact_updates):
        records = [record("q1")] if records is None else records
        scores = [score("q1")] if scores is None else scores
        run_path = self.root / (name + " run.jsonl")
        score_path = self.root / (name + " score.json")
        run_path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        artifact = {"schema": comparison.SCORE_SCHEMA, "bench": "locomo_plus",
                    "run_file": str(run_path), "results": scores,
                    "run": {"variant": "base", "benchmark": "locomo_plus", "run_id": "synthetic-run"}}
        artifact.update(artifact_updates)
        score_path.write_text(json.dumps(artifact), encoding="utf-8")
        return {"name": name, "run": run_path, "score": score_path}

    def compare(self, left, right=None, **kwargs):
        return comparison.compare_arms([left, right or self.arm("reference")], **kwargs)

    def reject(self, arm, pattern, **kwargs):
        with self.assertRaisesRegex(comparison.ComparisonError, pattern):
            self.compare(arm, **kwargs)

    def test_historical_120_scores_for_15_answers_regression(self):
        runs = [record(f"q{i}") for i in range(15)]
        rows = [score(f"q{i}", 1.0 if i < 8 else 0.0,
                      flag="eval_error" if i == 14 else None) for i in range(15)]
        rows += [score(f"q{i}", 0.0, flag="missing_record") for i in range(15, 120)]
        candidate = self.arm("candidate", runs, rows, overall={"n": 120, "mean_score": 0.0667})
        baseline = self.arm("baseline", runs, [score(f"q{i}", 0.0) for i in range(15)])
        report = self.compare(baseline, candidate)
        arm = report["arms"][1]
        self.assertEqual(arm["coverage"]["score_rows"], 120)
        self.assertEqual(arm["all_attempted"]["denominator"], 15)
        self.assertAlmostEqual(arm["all_attempted"]["mean"], 8 / 15)
        self.assertEqual(arm["judge_success"]["denominator"], 14)
        self.assertAlmostEqual(arm["judge_success"]["mean"], 8 / 14)
        self.assertEqual(arm["judge_flagged_qids"], ["q14"])
        self.assertEqual(len(arm["coverage"]["unattempted_score_qids"]), 105)
        pair = report["pairs"][0]
        self.assertEqual(pair["all_scored"]["paired_n"], 15)
        self.assertAlmostEqual(pair["all_scored"]["right"]["mean"], 8 / 15)
        self.assertEqual((pair["all_scored"]["up"], pair["all_scored"]["down"],
                          pair["all_scored"]["tie"]), (8, 0, 7))
        self.assertEqual(pair["judge_success"]["paired_n"], 14)

    def test_common_qids_come_from_answers_not_score_universe(self):
        left = self.arm("left", [record("a"), record("b")],
                        [score("a", 0), score("b", 1), score("c", 0, flag="missing_record")])
        right = self.arm("right", [record("b"), record("c")],
                         [score("a", 0, flag="missing_record"), score("b", 0.5), score("c", 1)])
        pair = self.compare(left, right)["pairs"][0]
        self.assertEqual(pair["common_answered_qids"], ["b"])
        self.assertEqual(pair["left_only_answered_qids"], ["a"])
        self.assertEqual(pair["right_only_answered_qids"], ["c"])
        self.assertEqual(pair["all_scored"]["mean_delta"], -0.5)
        self.assertEqual(pair["all_scored"]["down"], 1)

    def test_driver_timeout_empty_and_judge_error_are_distinct(self):
        runs = [record("ok"), record("timeout", answer_text=None, error="question timed out after 5s"),
                record("driver", error="provider failed"), record("empty", answer_text=" "),
                record("judge")]
        rows = [score("ok"), score("timeout", 0, flag="driver_error"),
                score("driver", 0, flag="driver_error"), score("empty", 0, flag="empty_answer"),
                score("judge", 0, flag="eval_error")]
        report = self.compare(self.arm("mixed", runs, rows))
        arm = report["arms"][0]
        self.assertEqual(arm["driver_status_counts"],
                         {"answered": 2, "timeout": 1, "driver_error": 1, "empty_answer": 1})
        self.assertEqual(arm["all_attempted"]["denominator"], 5)
        self.assertEqual(arm["all_attempted"]["mean"], 0.2)
        self.assertEqual(arm["judge_success"]["mean"], 1)
        self.assertEqual(arm["judge_flagged_qids"], ["judge"])
        self.assertEqual(arm["coverage"]["answered_qids"], ["judge", "ok"])

    def test_timeout_stop_reason_without_error(self):
        arm = self.arm("timeout", [record("q1", stop_reason="timeout")],
                       [score("q1", 0, flag="driver_error")])
        result = self.compare(arm)["arms"][0]
        self.assertEqual(result["driver_status_by_qid"], {"q1": "timeout"})
        self.assertEqual(result["judge_success"]["denominator"], 0)

    def test_missing_scores_are_not_imputed(self):
        left = self.arm("left", [record("a"), record("b")], [score("a")])
        right = self.arm("right", [record("a"), record("b")], [score("a", 0), score("b")])
        report = self.compare(left, right)
        metric = report["arms"][0]["all_attempted"]
        self.assertEqual(metric["denominator"], 2)
        self.assertEqual(metric["scored_n"], 1)
        self.assertIsNone(metric["mean"])
        self.assertEqual(metric["available_score_mean"], 1)
        self.assertEqual(metric["missing_score_qids"], ["b"])
        pair = report["pairs"][0]["all_scored"]
        self.assertIsNone(pair["mean_delta"])
        self.assertIsNone(pair["left"]["mean"])
        self.assertEqual(pair["paired_n"], 1)
        self.assertEqual(pair["available_mean_delta"], -1)

    def test_missing_record_placeholder_on_attempt_is_coverage_failure(self):
        arm = self.arm("stale", scores=[score("q1", 0, flag="missing_record")])
        result = self.compare(arm)["arms"][0]
        self.assertIsNone(result["all_attempted"]["mean"])
        self.assertEqual(result["coverage"]["missing_score_qids"], ["q1"])
        self.assertEqual(result["judge_flagged_qids"], [])
        self.assertEqual(len(result["coverage_issues"]), 1)

    def test_all_flags_and_explicit_judge_error_exclude_sensitivity(self):
        for updates in ({"flag": "parse_error"}, {"judge_error": "synthetic failure"},
                        {"judge_flag": "ambiguous"}, {"eval_error": "failure"}, {"error": "failure"}):
            with self.subTest(updates=updates):
                arm = self.arm("flagged", scores=[score("q1", 0.5, **updates)])
                report = self.compare(arm)
                self.assertEqual(report["arms"][0]["all_attempted"]["mean"], 0.5)
                self.assertIsNone(report["arms"][0]["judge_success"]["mean"])
                self.assertEqual(report["pairs"][0]["judge_success_excluded_qids"], ["q1"])

    def test_deterministic_success_and_evaluator_errors(self):
        runs = [record("q1", eval={"type": "f1"}), record("q2", eval={"type": "f1"})]
        rows = [score("q1", 0.5, eval_type="f1"), score("q2", 0, eval_type="f1", flag="eval_error")]
        arm = self.arm("deterministic", runs, rows)
        other = self.arm("other", [], [])
        result = self.compare(arm, other)["arms"][0]
        self.assertEqual(result["judge_success"]["qids"], ["q1"])
        self.assertEqual(result["evaluator_flagged_qids"], ["q2"])
        self.assertEqual(result["judge_flagged_qids"], [])

    def test_duplicates_fail_by_default_in_either_input(self):
        for source in ("run", "score"):
            for divergent in (False, True):
                with self.subTest(source=source, divergent=divergent):
                    runs = [record("q1")]
                    rows = [score("q1")]
                    if source == "run":
                        runs.append(record("q1", answer_text="other" if divergent else "synthetic answer"))
                    else:
                        rows.append(score("q1", 0 if divergent else 1))
                    self.reject(self.arm("duplicate", runs, rows), "duplicate qid")

    def test_explicit_identical_and_first_last_dedup(self):
        identical = self.arm("identical", [record("q1"), record("q1")], [score("q1"), score("q1")])
        result = self.compare(identical, dedup="identical")["arms"][0]
        self.assertEqual(result["all_attempted"]["denominator"], 1)
        self.assertFalse(result["duplicates"]["run"][0]["divergent"])
        divergent = self.arm("divergent", [record("q1"), record("q1", answer_text="changed")],
                             [score("q1", 0), score("q1", 1)])
        self.reject(divergent, "divergent duplicate", dedup="identical")
        for policy, value, position in (("first", 0, 1), ("last", 1, 2)):
            result = self.compare(divergent, dedup=policy)["arms"][0]
            self.assertEqual(result["all_attempted"]["mean"], value)
            self.assertTrue(result["duplicates"]["run"][0]["divergent"])
            self.assertEqual(result["duplicates"]["score"][0]["kept_position"], position)

    def test_mixed_variants_need_explicit_selection(self):
        arm = self.arm("mixed", [record("q1"), record("q1", variant="nomem")])
        self.reject(arm, "ambiguous run variants")
        arm["variant"] = "base"
        result = self.compare(arm)["arms"][0]
        self.assertEqual(result["coverage"]["excluded_variant_records"], 1)
        self.assertEqual(result["duplicates"]["run"], [])
        arm["variant"] = "nomem"
        self.reject(arm, "variant mismatch")
        arm["variant"] = "absent"
        self.reject(arm, "no run records")

    def test_different_variants_across_arms_are_allowed(self):
        left = self.arm("left")
        right = self.arm("right", [record("q1", variant="nomem")], run={"variant": "nomem"})
        self.assertEqual(self.compare(left, right)["pairs"][0]["all_scored"]["tie"], 1)

    def test_score_row_and_metadata_variant_mismatches(self):
        self.reject(self.arm("wrong", scores=[score("q1", variant="nomem")]), "variant mismatch")
        arm = self.arm("mixed", [record("q1", variant=None)],
                       [score("q1", variant="base"), score("q2", variant="nomem")], run={})
        self.reject(arm, "mixed score row variants")

    def test_metadata_absence_is_not_invented_verification(self):
        bare = self.arm("bare", [{"qid": "q1", "answer_text": "answer"}],
                        [{"qid": "q1", "score": 1}], run={}, bench=None)
        result = self.compare(bare)["arms"][0]
        self.assertIsNone(result["variant"])
        self.assertIsNone(result["lineage"]["score_declared_variant"])
        self.assertIsNone(result["lineage"]["score_declared_run_sha256"])
        self.assertIn("no run-content hash", result["lineage"]["binding"])

    def test_score_run_sha256_verifies_exact_full_input_bytes_without_metric_changes(self):
        arm = self.arm("bound", [record("q1"), record("q1", variant="nomem")])
        arm["variant"] = "base"
        run_path, score_path = Path(arm["run"]), Path(arm["score"])
        raw = b"\xef\xbb\xbf" + run_path.read_bytes().replace(b"\n", b"\r\n")
        run_path.write_bytes(raw)
        digest = hashlib.sha256(raw).hexdigest()
        unbound = self.compare(arm)
        artifact = json.loads(score_path.read_text())
        for declared in (digest, digest.upper()):
            with self.subTest(declared=declared):
                artifact["run_sha256"] = declared
                score_path.write_text(json.dumps(artifact), encoding="utf-8")
                report = self.compare(arm)
                lineage = report["arms"][0]["lineage"]
                self.assertEqual(lineage["score_declared_run_sha256"], declared)
                self.assertEqual(lineage["run"]["sha256"], digest)
                self.assertEqual(lineage["binding"], "verified run-content SHA-256")
                self.assertEqual(report["arms"][0]["coverage"]["excluded_variant_records"], 1)
                for field in ("coverage", "all_attempted", "judge_success"):
                    self.assertEqual(report["arms"][0][field], unbound["arms"][0][field])
                self.assertEqual(report["pairs"], unbound["pairs"])

    def test_score_run_sha256_rejects_changed_raw_bytes_even_if_records_are_unchanged(self):
        arm = self.arm("stale-hash")
        run_path, score_path = Path(arm["run"]), Path(arm["score"])
        artifact = json.loads(score_path.read_text())
        artifact["run_sha256"] = hashlib.sha256(run_path.read_bytes()).hexdigest()
        score_path.write_text(json.dumps(artifact), encoding="utf-8")
        run_path.write_bytes(run_path.read_bytes() + b"\n")
        self.reject(arm, "run_sha256 mismatch")

    def test_score_run_sha256_rejects_malformed_present_fields(self):
        for declared in (None, True, 7, [], {}, "", "a" * 63, "a" * 65,
                         "g" * 64, " " + "a" * 64, "a" * 64 + "\n"):
            with self.subTest(declared=declared):
                self.reject(self.arm("bad-hash", run_sha256=declared),
                            "run_sha256 must be exactly 64 hexadecimal characters")

    def test_merged_run_ids_allow_sample_metadata_but_reject_unrelated_id(self):
        runs = [record("q1", run_id="shard-a"), record("q2", run_id="shard-b")]
        arm = self.arm("merged", runs, [score("q1"), score("q2")], run={"run_id": "shard-a"})
        self.assertEqual(self.compare(arm)["arms"][0]["lineage"]["run_ids"], ["shard-a", "shard-b"])
        self.reject(self.arm("wrong", runs, run={"run_id": "unrelated"}), "run_id")

    def test_benchmark_and_qid_identity_mismatches_fail(self):
        for updates in ({"env_id": "different"}, {"question": "Different?"},
                        {"eval": {"type": "llm_judge", "params": {"x": 1}}}, {"answer": "different"}):
            with self.subTest(updates=updates):
                left = self.arm("left", [record("q1", answer="gold")])
                right = self.arm("right", [record("q1", **updates)],
                                 [score("q1", env_id=updates.get("env_id", "synthetic-env"))])
                with self.assertRaisesRegex(comparison.ComparisonError, "cross-arm qid"):
                    self.compare(left, right)
        self.reject(self.arm("wrong", bench="locomo"), "benchmark mismatch")
        left = self.arm("left")
        right = self.arm("right", [record("q1", benchmark="locomo")], bench="locomo", run={})
        with self.assertRaisesRegex(comparison.ComparisonError, "cross-arm benchmark"):
            self.compare(left, right)

    def test_score_row_identity_and_driver_contradictions_fail(self):
        for updates, pattern in (({"env_id": "other"}, "env_id"),
                                 ({"eval_type": "f1"}, "eval_type"),
                                 ({"run_id": "other"}, "run_id"),
                                 ({"flag": "driver_error"}, "contradicts"),
                                 ({"flag": "empty_answer"}, "contradicts")):
            with self.subTest(updates=updates):
                self.reject(self.arm("wrong", scores=[score("q1", 0, **updates)]), pattern)
        self.reject(self.arm("wrong", [record("q1", error="failure")]), "contradicts")

    def test_invalid_scores_fail_even_for_unattempted_or_dedup_discarded_rows(self):
        for value in (-0.1, 1.1, True, "0.5", None, float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                self.reject(self.arm("bad", scores=[score("q1"), score("extra", value)]),
                            "finite|score must")
        arm = self.arm("bad", scores=[score("q1", -1), score("q1")])
        self.reject(arm, "score must", dedup="last")
        arm = self.arm("bad")
        path = Path(arm["score"])
        path.write_text(path.read_text().replace('"score": 1.0', '"score": 1e999'), encoding="utf-8")
        self.reject(arm, "non-finite")

    def test_malformed_json_and_schema_fail(self):
        for content, pattern in (('{"qid":"q1","qid":"q2"}', "duplicate JSON"),
                                 ('{"qid":', "run:1"), ('[]', "expected an object"),
                                 ('{"qid":17}', "qid must"), ('{}', "qid must")):
            with self.subTest(content=content):
                arm = self.arm("malformed")
                Path(arm["run"]).write_text(content, encoding="utf-8")
                self.reject(arm, pattern)
        for updates in ({"schema": "unknown"}, {"results": {}}, {"run": []}):
            with self.subTest(updates=updates):
                self.reject(self.arm("schema", **updates), "expected|must be")
        self.reject(self.arm("answer", [record("q1", answer_text=42)]), "answer_text")

    def test_empty_and_disjoint_runs_have_null_means(self):
        report = self.compare(self.arm("empty", [], []))
        self.assertIsNone(report["arms"][0]["all_attempted"]["mean"])
        self.assertEqual(report["pairs"][0]["all_scored"]["paired_n"], 0)
        self.assertIsNone(report["pairs"][0]["all_scored"]["mean_delta"])
        report = self.compare(self.arm("disjoint", [record("other")], [score("other")]))
        self.assertEqual(report["common_answered_qids"], [])

    def test_three_arms_have_pairwise_and_global_intersections(self):
        arms = [self.arm(name, [record(qid) for qid in qids], [score(qid) for qid in qids])
                for name, qids in (("a", ["a", "b"]), ("b", ["b", "c"]), ("c", ["a", "c"]))]
        report = comparison.compare_arms(arms)
        self.assertEqual(len(report["pairs"]), 3)
        self.assertEqual(report["common_answered_qids"], [])
        self.assertTrue(all(len(pair["common_answered_qids"]) == 1 for pair in report["pairs"]))

    def test_sha256_uses_exact_bytes_and_report_is_deterministic(self):
        left, right = self.arm("left"), self.arm("right")
        Path(left["run"]).write_bytes(b"\xef\xbb\xbf" + Path(left["run"]).read_bytes() + b"\r\n")
        report = self.compare(left, right)
        for key in ("run", "score"):
            lineage = report["arms"][0]["lineage"][key]
            raw = Path(left[key]).read_bytes()
            self.assertEqual(lineage["sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(lineage["bytes"], len(raw))
        self.assertEqual(report, self.compare(left, right))
        json.dumps(report, allow_nan=False)

    def test_requires_unique_explicit_arms_and_valid_policy(self):
        arm = self.arm("one")
        with self.assertRaisesRegex(comparison.ComparisonError, "at least two"):
            comparison.compare_arms([arm])
        with self.assertRaisesRegex(comparison.ComparisonError, "unique"):
            comparison.compare_arms([arm, arm])
        self.reject(arm, "unknown dedup", dedup="best")

    def cli(self, argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = comparison.main(argv)
        return code, out.getvalue(), err.getvalue()

    def cli_args(self):
        left, right = self.arm("left"), self.arm("right")
        args = ["--arm", "left", str(left["run"]), str(left["score"]),
                "--arm", "right", str(right["run"]), str(right["score"])]
        return args, left, right

    def test_cli_stdout_output_file_and_variants(self):
        args, _, _ = self.cli_args()
        code, out, err = self.cli(args + ["--variant", "left", "base"])
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(json.loads(out)["schema"], comparison.SCHEMA)
        path = self.root / "report.json"
        self.assertEqual(self.cli(args + ["--out", str(path)]), (0, "", ""))
        self.assertEqual(json.loads(path.read_text())["arms"][0]["name"], "left")

    def test_cli_rejects_bad_variant_and_input_overwrite(self):
        args, left, _ = self.cli_args()
        for options in (["--variant", "missing", "base"],
                        ["--variant", "left", "base", "--variant", "left", "base"],
                        ["--out", str(left["run"])]):
            with self.subTest(options=options):
                code, out, err = self.cli(args + options)
                self.assertEqual((code, out), (2, ""))
                self.assertIn("error", json.loads(err))
        self.assertIn("qid", Path(left["run"]).read_text())

    def test_cli_rejects_hardlinked_output_without_changing_any_input(self):
        for name in ("left", "right"):
            for key in ("run", "score"):
                with self.subTest(arm=name, input=key):
                    args, left, right = self.cli_args()
                    arms = {"left": left, "right": right}
                    inputs = [Path(arm[field]) for arm in arms.values()
                              for field in ("run", "score")]
                    before = {path: path.read_bytes() for path in inputs}
                    output = self.root / f"hardlink-{name}-{key}.json"
                    target = Path(arms[name][key])
                    output.hardlink_to(target)
                    self.assertTrue(output.samefile(target))
                    code, out, err = self.cli(args + ["--out", str(output)])
                    self.assertEqual((code, out), (2, ""))
                    self.assertIn("output cannot overwrite an input file", json.loads(err)["error"])
                    self.assertEqual({path: path.read_bytes() for path in inputs}, before)
                    self.assertEqual(output.read_bytes(), before[target])

    def test_cli_output_identity_inspection_errors_fail_without_writes(self):
        args, left, right = self.cli_args()
        inputs = [Path(arm[key]) for arm in (left, right) for key in ("run", "score")]
        before = {path: path.read_bytes() for path in inputs}
        output = self.root / "existing-report.json"
        output.write_bytes(b"keep existing report")
        for error in (PermissionError("synthetic identity access denied"),
                      OSError("synthetic identity inspection failed"),
                      FileNotFoundError("synthetic file disappeared during inspection")):
            with self.subTest(error=type(error).__name__):
                with mock.patch.object(Path, "samefile", side_effect=error) as inspected:
                    code, out, err = self.cli(args + ["--out", str(output)])
                self.assertTrue(inspected.called)
                self.assertEqual((code, out), (2, ""))
                self.assertIn(str(error), json.loads(err)["error"])
                self.assertEqual(output.read_bytes(), b"keep existing report")
                self.assertEqual({path: path.read_bytes() for path in inputs}, before)

    def test_cli_output_stat_error_fails_without_writes(self):
        args, left, right = self.cli_args()
        inputs = [Path(arm[key]) for arm in (left, right) for key in ("run", "score")]
        before = {path: path.read_bytes() for path in inputs}
        output = self.root / "existing-report.json"
        output.write_bytes(b"keep existing report")
        original_stat = Path.stat

        def denied_stat(path, *args, **kwargs):
            if path == output:
                raise PermissionError("synthetic output stat denied")
            return original_stat(path, *args, **kwargs)

        with mock.patch.object(Path, "stat", denied_stat):
            code, out, err = self.cli(args + ["--out", str(output)])
        self.assertEqual((code, out), (2, ""))
        self.assertIn("synthetic output stat denied", json.loads(err)["error"])
        self.assertEqual(output.read_bytes(), b"keep existing report")
        self.assertEqual({path: path.read_bytes() for path in inputs}, before)

    def test_cli_existing_unrelated_output_remains_supported(self):
        args, left, right = self.cli_args()
        inputs = [Path(arm[key]) for arm in (left, right) for key in ("run", "score")]
        before = {path: path.read_bytes() for path in inputs}
        output = self.root / "existing-report.json"
        output.write_bytes(b"old report")
        self.assertEqual(self.cli(args + ["--out", str(output)]), (0, "", ""))
        self.assertEqual(json.loads(output.read_text())["schema"], comparison.SCHEMA)
        self.assertEqual({path: path.read_bytes() for path in inputs}, before)

    def test_cli_validation_failure_does_not_write_partial_report(self):
        args, left, _ = self.cli_args()
        Path(left["score"]).write_text("invalid", encoding="utf-8")
        path = self.root / "report.json"
        path.write_text("keep", encoding="utf-8")
        code, out, err = self.cli(args + ["--out", str(path)])
        self.assertEqual((code, out), (2, ""))
        self.assertIn("error", json.loads(err))
        self.assertEqual(path.read_text(), "keep")

    def test_cli_score_run_sha256_mismatch_leaves_existing_report_unchanged(self):
        args, left, right = self.cli_args()
        score_path = Path(left["score"])
        artifact = json.loads(score_path.read_text())
        artifact["run_sha256"] = hashlib.sha256(b"another synthetic run").hexdigest()
        score_path.write_text(json.dumps(artifact), encoding="utf-8")
        inputs = [Path(arm[key]) for arm in (left, right) for key in ("run", "score")]
        before = {path: path.read_bytes() for path in inputs}
        output = self.root / "report.json"
        output.write_bytes(b"keep report")
        code, out, err = self.cli(args + ["--out", str(output)])
        self.assertEqual((code, out), (2, ""))
        self.assertIn("run_sha256 mismatch", json.loads(err)["error"])
        self.assertEqual(output.read_bytes(), b"keep report")
        self.assertEqual({path: path.read_bytes() for path in inputs}, before)

    def test_cli_missing_file_is_json_error(self):
        args, left, _ = self.cli_args()
        Path(left["run"]).unlink()
        code, out, err = self.cli(args)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("error", json.loads(err))

    def test_help_documents_semantics_and_explicit_inputs(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), self.assertRaises(SystemExit) as caught:
            comparison.main(["--help"])
        self.assertEqual(caught.exception.code, 0)
        for text in ("--arm NAME RUN_JSONL SCORE_JSON", "--dedup", "--variant", "never imputed"):
            self.assertIn(text, out.getvalue())


if __name__ == "__main__":
    unittest.main()
