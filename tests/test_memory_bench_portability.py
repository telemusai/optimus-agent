"""Offline portability checks for the isolated memory benchmark package."""
from __future__ import annotations

import ast
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench.converters import common, lme_v2, locomo, locomo_plus, longmemeval
from bench.eval import deterministic as det
from bench.eval import evaluate, judges, local_scoring, local_adapters, porter
from bench.protocol import annotate_question, EVALUATOR_VERSION, evaluator_identity
from bench import evidence
from bench.tests import test_local_scoring as original_scoring_tests

FIXTURES = ROOT / "bench" / "fixtures"


class MemoryBenchPortabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="memory-bench-portable-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = dict(os.environ)
        for key in ("MEMORY_BENCH_MODELS_JSON", "MEMORY_BENCH_JUDGE_CACHE",
                    "MEMORY_BENCH_DATA_ROOT", "MEMORY_BENCH_COMMON_ROOT",
                    "LP_JUDGE_MAX_TOKENS", "LME_V2_AXTREE_CAP"):
            self.env.pop(key, None)
        self.env["PYTHONDONTWRITEBYTECODE"] = "1"
        self.env["PYTHONPATH"] = str(ROOT)
        self.env["HOME"] = str(self.root / "home")
        self.env["USERPROFILE"] = self.env["HOME"]
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, self.env, clear=True).start()
        self.old_client = judges._shared_client
        self.addCleanup(judges.set_client, self.old_client)
        judges.set_client(None)
        self.old_roots = common.DATA_ROOT, common.COMMON_ROOT
        self.addCleanup(common.configure_paths, *self.old_roots)

    def cli(self, module, *args, ok=True, env=None):
        result = subprocess.run(
            [sys.executable, "-B", "-m", module, *map(str, args)],
            cwd=ROOT, env=env or self.env, capture_output=True, text=True, timeout=30,
        )
        if ok:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def dump(self, name, value, jsonl=False):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if jsonl and Path(name).name.startswith("qs_"):
            value = [annotate_question(dict(row, eval=dict(row["eval"])), row.get("family", "locomo")) for row in value]
        data = ("".join(json.dumps(r) + "\n" for r in value) if jsonl
                else json.dumps(value))
        path.write_text(data, encoding="utf-8")
        return path

    def config(self, *, api_key="synthetic-api-key-123", header="synthetic-session-456"):
        return self.dump("explicit-models.json", {"providers": {"dgx-glm53": {
            "baseUrl": "https://dgx.fixture.invalid/v1", "apiKey": api_key,
            "headers": {"X-Pomerium-Authorization": header},
        }}})

    def convert(self):
        output = self.root / "common with spaces"
        self.cli("bench.converters.run_converters", "--data-root", FIXTURES / "raw",
                 "--out-dir", output, "--full")
        return output

    def test_imports_have_no_network_process_or_credential_side_effects(self):
        script = textwrap.dedent("""
            import importlib, pkgutil, pathlib, socket, subprocess
            from unittest.mock import patch
            import bench
            modules = [m.name for m in pkgutil.walk_packages(bench.__path__, "bench.")]
            def forbidden(*args, **kwargs):
                raise AssertionError("unexpected import side effect")
            with patch.object(pathlib.Path, "read_text", forbidden), \
                 patch.object(pathlib.Path, "mkdir", forbidden), \
                 patch.object(socket, "create_connection", forbidden), \
                 patch.object(subprocess, "run", forbidden), \
                 patch.object(subprocess, "Popen", forbidden):
                for name in modules:
                    importlib.import_module(name)
                from bench.eval.judges import DGXClient
                DGXClient()
        """)
        result = subprocess.run([sys.executable, "-B", "-c", script], cwd=ROOT,
                                env=self.env, capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertFalse((self.root / "home").exists())

    def test_cli_help_does_not_read_config(self):
        env = dict(self.env, MEMORY_BENCH_MODELS_JSON=str(self.root / "nonexistent.json"))
        for module in ("bench.converters.run_converters", "bench.eval.evaluate",
                       "bench.analysis.attribute", "bench.prepare_questions"):
            with self.subTest(module=module):
                self.assertIn("usage:", self.cli(module, "--help", env=env).stdout)

    def test_all_four_converters_and_full_manifests_are_portable(self):
        output = self.convert()
        expected = {"longmemeval": (1, 1), "locomo": (1, 5), "lme_v2": (2, 2),
                    "locomo_plus": (3, 3)}
        for family, (envs, questions) in expected.items():
            with self.subTest(family=family):
                manifest = json.loads((output / family / "manifest.json").read_text())
                self.assertEqual(manifest["n_envs"], envs)
                self.assertEqual(manifest["n_questions"], questions)
                self.assertEqual(manifest["seed"], 20261007)
                self.assertTrue((output / family / "full" / "manifest.json").is_file())
                qs = evaluate.load_questions(family, output / family)
                full = evaluate.load_questions(family, output / family / "full")
                self.assertEqual(len(qs), questions)
                self.assertEqual(len(full), questions)
                for q in qs:
                    self.assertIsInstance(q["qid"], str)
                    self.assertIn("eval", q)
                for record in manifest["envs"]:
                    self.assertTrue((output / family / record["file"]).is_file())
        lme_env = json.loads((output / "longmemeval/env_synthetic_lme.json").read_text())
        self.assertEqual([s["session_id"] for s in lme_env["sessions"]], ["s1", "s2"])
        loco_qs = evaluate.load_questions("locomo", output / "locomo")
        self.assertNotIn("presented_question", loco_qs[1])
        self.assertEqual(loco_qs[1]["category_name"], "temporal")
        self.assertEqual(loco_qs[4]["eval"]["type"], "abstain_f1")
        v2_env = json.loads((output / "lme_v2/env_web.json").read_text())
        self.assertEqual(v2_env["ground_truth_refs"], [])
        plus_env = json.loads((output / "locomo_plus/env_synthetic_cue_0.json").read_text())
        self.assertEqual(plus_env["ground_truth_refs"], ["synthetic-cue"])

    def test_converter_environment_roots_and_determinism(self):
        out_a, out_b = self.root / "a", self.root / "b"
        env = dict(self.env, MEMORY_BENCH_DATA_ROOT=str(FIXTURES / "raw"),
                   MEMORY_BENCH_COMMON_ROOT=str(out_a))
        self.cli("bench.converters.run_converters", "--family", "locomo", env=env)
        self.cli("bench.converters.run_converters", "--family", "locomo",
                 "--data-root", FIXTURES / "raw", "--out-dir", out_b)
        files_a = {p.relative_to(out_a): p.read_bytes() for p in out_a.rglob("*") if p.is_file()}
        files_b = {p.relative_to(out_b): p.read_bytes() for p in out_b.rglob("*") if p.is_file()}
        self.assertEqual(files_a, files_b)

    def test_converter_outputs_cannot_escape_configured_root(self):
        common.configure_paths(FIXTURES / "raw", self.root / "allowed")
        (self.root / "allowed").mkdir()
        for write in (common.write_json, common.write_jsonl):
            with self.assertRaisesRegex(ValueError, "escapes"):
                write(str(self.root / "outside.json"), [])
        self.assertFalse((self.root / "outside.json").exists())
        with self.assertRaisesRegex(ValueError, "escapes"):
            common.family_dir("../../escape")
        with self.assertRaisesRegex(ValueError, "escapes"):
            common.clean_outputs(str(self.root))

    def test_documented_offline_eval_cli_and_attribution(self):
        output = self.convert()
        scores_file = self.root / "scores.json"
        result = self.cli("bench.eval.evaluate", "--run", FIXTURES / "sample-run.jsonl",
                          "--bench", "locomo", "--qs-dir", output / "locomo",
                          "--out", scores_file, "--variant", "base")
        scores = json.loads(scores_file.read_text())
        self.assertEqual(scores["schema"], "memory-bench-scores/1")
        self.assertEqual(scores["overall"]["mean_score"], 1.0)
        self.assertEqual(scores["overall"]["n_flagged"], 0)
        self.assertIsNone(scores["judge_stats"])
        self.assertEqual(scores["run_sha256"], hashlib.sha256((FIXTURES / "sample-run.jsonl").read_bytes()).hexdigest())
        envs = self.dump("envs.jsonl", [{"env_id": "synthetic_conv", "variant": "base",
            "benchmark": "locomo", "run_id": "synthetic-declaration-v1",
            "store_snapshot": {"memory": {"m1": {"title": "Bicycle", "content": "blue bicycle green 9 January"}}}}], True)
        attr = self.root / "attribution.json"
        self.cli("bench.analysis.attribute", "--runs", FIXTURES / "sample-run.jsonl",
                 "--envs", envs, "--bench", "locomo", "--scores", scores_file,
                 "--data-common", output, "--variant", "base", "--out", attr)
        report = json.loads(attr.read_text())
        self.assertEqual(report["questions"], 5)
        self.assertEqual(report["abstention"], {"threshold-met": 1})
        self.assertEqual(report["binding_status"], "verified-local-artifact-consistency")
        self.assertEqual(len(report["per_question"]), 5)
        self.assertIn("5 questions scored", result.stdout)

    def test_strict_variant_filter_last_record_wins_and_answer_text(self):
        q = annotate_question({"qid": "q", "env_id": "e", "category": "4", "answer": "blue", "eval": {"type": "f1"}}, "locomo")
        rows = [{"qid": "q", "variant": "base", "answer_text": "blue"},
                {"qid": "q", "variant": "nomem", "answer_text": "blue"},
                {"qid": "q", "variant": "base", "answer_text": "red", "answer": "blue"}]
        path = self.dump("run.jsonl", rows, True)
        records, dupes = evaluate.load_run(path, "base")
        self.assertEqual(dupes, 1)
        self.assertEqual(records["q"]["answer_text"], "red")
        self.assertEqual(evaluate.evaluate_question(q, records["q"])["score"], 0)
        self.assertEqual(evaluate.load_run(path, "absent"), ({}, 0))

    def test_legacy_flags_denominators_and_native_failures_stay_separate(self):
        q = annotate_question({"qid": "q", "category": "4", "answer": "blue", "eval": {"type": "f1"}}, "locomo")
        result = [evaluate.evaluate_question(q, None),
                  evaluate.evaluate_question(q, {"error": "secret-provider-error", "answer_text": "blue"}),
                  evaluate.evaluate_question(q, {"answer_text": " "}),
                  evaluate.evaluate_question(dict(q, eval=dict(q["eval"], type="not-a-metric")), {"answer_text": "blue"}),
                  evaluate.evaluate_question(q, {"answer_text": "blue", "stop_reason": "error", "timed_out": True})]
        self.assertEqual([r["flag"] for r in result],
                         ["missing_record", "driver_error", "empty_answer", "unknown_eval_type", None])
        self.assertEqual(result[-1]["score"], 1)
        self.assertEqual(result[-1]["native_failure"], {"driver_error": False, "stop_error": True, "timed_out": True})
        agg = evaluate.aggregate(result)
        self.assertEqual((agg["n"], agg["n_flagged"], agg["mean_score"]), (5, 4, 0.2))
        self.assertNotIn("secret-provider-error", json.dumps(result))

    def test_deterministic_metric_golden_vectors(self):
        self.assertEqual(local_scoring.lexical_f1("running dogs", "run dog", porter.porter_stem), 1)
        self.assertAlmostEqual(local_scoring.mean_best_match(["blue"], ["blue", "green"]), 0.5)
        self.assertTrue(local_scoring.token_set_equal("the blue blue bicycle", "bicycle blue"))
        params = {"option_map": {"a": "cat", "b": "Insufficient information."}, "abstention_label": "b"}
        self.assertEqual(local_adapters.score_local("abstain_f1", params, "(a)", "")["score"], 0)
        self.assertEqual(local_adapters.score_local("abstain_f1", params, "(b)", "")["score"], 1)
        self.assertTrue(det.norm_phrase_set_match("green blue", "blue;green"))
        self.assertFalse(det.norm_phrase_set_match_ordered("green blue", "blue;green"))
        self.assertTrue(det.mc_choice_match("Option A.", "A"))
        self.assertTrue(det.mc_choice_set_match("A and B", "B,A"))
        self.assertEqual(evaluate.wilson_ci(0, 0), (None, None))
        self.assertEqual(evaluate.wilson_ci(5, 10), (0.2366, 0.7634))

    def test_lme_v2_boxed_unknown_rule_is_unchanged(self):
        q = annotate_question({"qid": "v2", "answer": "UNKNOWN", "eval": {"type": "phrase_set"}}, "lme_v2")
        out = evaluate.evaluate_question(q, {"answer_text": r"Earlier \boxed{blue}. Final \boxed{UNKNOWN}"})
        self.assertEqual(out["parsed_boxed"], "UNKNOWN")
        self.assertTrue(out["is_unknown"])
        self.assertTrue(out["unknown_forced_zero"])
        self.assertEqual(out["score"], 0)
        self.assertEqual(det.extract_boxed_answer(r"\boxed{a{b}c}"), "a{b}c")

    def test_no_config_or_imported_profile_is_required_for_cache_only_client(self):
        with patch.object(Path, "read_text", side_effect=AssertionError("implicit read")), \
             patch.object(Path, "mkdir", side_effect=AssertionError("implicit write")):
            client = judges.DGXClient()
        with patch.object(client, "_default_transport", side_effect=AssertionError("network")):
            with self.assertRaisesRegex(RuntimeError, "network disabled"):
                client.chat_cached("longmemeval", [{"role": "user", "content": "fixture"}])
        self.assertEqual(client.stats["requests"], 0)

    def test_explicit_configuration_does_not_enable_network(self):
        with patch.object(judges.DGXClient, "_default_transport", side_effect=AssertionError("network")) as transport:
            client = judges.DGXClient(models_json=self.config())
            with self.assertRaisesRegex(RuntimeError, "network disabled"):
                client.chat_cached("longmemeval", [])
            transport.assert_not_called()
        with self.assertRaisesRegex(ValueError, "explicit"):
            judges.DGXClient(allow_network=True)

    def test_command_credentials_are_rejected_not_executed(self):
        with patch.object(subprocess, "run", side_effect=AssertionError("credential command")) as run:
            with self.assertRaisesRegex(ValueError, "commands are not supported"):
                judges._resolve_secret("!echo synthetic-secret")
            with self.assertRaisesRegex(ValueError, "invalid explicit"):
                judges.DGXClient(models_json=self.config(api_key="!echo synthetic-secret"))
            run.assert_not_called()

    def test_explicit_environment_secret_resolution_and_model_constraint(self):
        os.environ["SYNTHETIC_BENCH_KEY"] = "synthetic-key-from-env"
        self.addCleanup(os.environ.pop, "SYNTHETIC_BENCH_KEY", None)
        client = judges.DGXClient(models_json=self.config(api_key={"env": "SYNTHETIC_BENCH_KEY"}))
        self.assertEqual(client.api_key, "synthetic-key-from-env")
        for kwargs in ({"provider": "other-provider"}, {"model": "other-model"}):
            with self.assertRaisesRegex(ValueError, "require DGX"):
                judges.DGXClient(**kwargs)

    def test_new_cache_namespace_is_offline_and_rejects_campaign_cache(self):
        cache = self.root / "cache"
        cache.mkdir()
        messages = [{"role": "user", "content": "synthetic prompt"}]
        key = {"judge": "longmemeval", "model": "glm-5.3", "temperature": 0,
               "max_tokens": 2048, "messages": messages}
        historical_digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        (cache / f"{historical_digest}.json").write_text(json.dumps({"content": "yes", "meta": {"latency_ms": 7}}))
        client = judges.DGXClient(cache_dir=cache)
        with self.assertRaisesRegex(RuntimeError, "network disabled"):
            client.chat_cached("longmemeval", messages, temperature=0, max_tokens=2048)
        key["protocol"] = EVALUATOR_VERSION
        digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        self.assertNotEqual(digest, historical_digest)
        (cache / f"{digest}.json").write_text(json.dumps({"protocol": EVALUATOR_VERSION,
            "model": "glm-5.3", "request_sha256": digest, "content": "yes", "meta": {"latency_ms": 7}}))
        with patch.object(client, "_post_chat", side_effect=AssertionError("network")):
            content, meta = client.chat_cached("longmemeval", messages, temperature=0, max_tokens=2048)
        self.assertEqual((content, meta["latency_ms"]), ("yes", 7))
        self.assertEqual(client.stats["cache_hits"], 1)
        self.assertEqual(len(list(cache.iterdir())), 2)

    def test_all_judge_adapters_work_with_synthetic_transport_and_redact_cache(self):
        calls = []
        def transport(url, headers, body, timeout):
            calls.append((url, headers, body, timeout))
            text = body["messages"][-1]["content"]
            if '"DATA"' in text:
                content = '{"correct":true,"reason":"synthetic-api-key-123 synthetic-session-456"}'
            elif "Output JSON only:" in text:
                content = '{"label":1,"reason":"synthetic-api-key-123 synthetic-session-456"}'
            else:
                content = "yes synthetic-api-key-123 synthetic-session-456"
            return 200, {"choices": [{"message": {"content": content}, "finish_reason": "stop"}], "usage": {"total_tokens": 3}}
        cache = self.root / "cache"
        client = judges.DGXClient(models_json=self.config(), transport=transport, cache_dir=cache)
        judges.set_client(client)
        responses = [judges.judge_longmemeval("color?", "blue", "blue", "single-session-user"),
                     judges.judge_locomo_plus("needs helmet", "buy helmet"),
                     judges.judge_lme_v2_abstention("flawed?", "no", "no", "no"),
                     judges.judge_lme_v2_gotchas("issue?", "insight", "insight", "insight")]
        self.assertEqual([r["score"] for r in responses], [1, 1, 1, 1])
        self.assertEqual([c[2]["max_tokens"] for c in calls], [2048, 512, 4096, 4096])
        self.assertTrue(all(c[2]["model"] == "glm-5.3" for c in calls))
        self.assertNotIn("temperature", calls[-1][2])
        for content in [json.dumps(responses), *[p.read_text() for p in cache.glob("*.json")]]:
            self.assertNotIn("synthetic-api-key-123", content)
            self.assertNotIn("synthetic-session-456", content)
            self.assertIn("[REDACTED]", content)
        self.assertEqual(len(list(cache.glob("*.json"))), 4)
        self.assertFalse(list(cache.glob("*.tmp")))

    def test_new_local_parser_rejects_historical_substring_credit(self):
        for response in ("incorrect", "correct", '{"label":"correct","reason":"x"}'):
            with self.assertRaises(local_scoring.JudgeResponseError):
                local_scoring.parse_judge_response(response)
        self.assertEqual(judges.v2_parse_llm_binary_judgement('```json\n{"label":1}\n```')[0], 1)

    def test_judge_failure_is_flagged_without_leaking_exception(self):
        def failing(*args):
            raise RuntimeError("synthetic-api-key-123 synthetic-session-456")
        client = judges.DGXClient(models_json=self.config(), transport=failing, retries=1)
        judges.set_client(client)
        q = annotate_question({"qid": "q", "question": "color?", "answer": "blue",
                               "eval": {"type": "llm_judge"}}, "longmemeval")
        result = evaluate.evaluate_question(q, {"answer_text": "blue"})
        self.assertEqual((result["flag"], result["score"]), ("eval_error", 0))
        self.assertNotIn("synthetic-api-key", json.dumps(result))
        self.assertNotIn("synthetic-session", json.dumps(result))

    def test_judge_cli_cache_miss_is_offline_and_invalid_config_is_redacted(self):
        qdir = self.root / "qs"
        self.dump("qs/qs_e.jsonl", [{"qid": "q", "env_id": "e", "question": "color?", "answer": "blue",
                                   "family": "longmemeval", "eval": {"type": "llm_judge"}}], True)
        question = evidence.read_questions(qdir)[0][0]
        runs = self.dump("run.jsonl", [self.invented_record(question, answer_text="blue")], True)
        scores = self.root / "scores.json"
        args = ("--run", runs, "--bench", "longmemeval", "--qs-dir", qdir, "--out", scores)
        self.cli("bench.eval.evaluate", *args)
        result = json.loads(scores.read_text())
        self.assertEqual(result["results"][0]["flag"], "eval_error")
        self.assertEqual(result["judge_stats"]["requests"], 0)
        bad = self.config(api_key="!echo synthetic-private-value")
        res = self.cli("bench.eval.evaluate", *args, "--models-json", bad, ok=False)
        self.assertEqual(res.returncode, 2)
        self.assertNotIn("synthetic-private-value", res.stdout + res.stderr)

    def test_stdlib_transport_disables_proxy_and_redirect_credentials(self):
        from urllib.request import ProxyHandler
        client = judges.DGXClient(models_json=self.config(), allow_network=True)
        handlers = []
        class Response:
            status = 200
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return None
            def read(self):
                return b'{"choices":[{"message":{"content":"yes"}}]}'
        class Opener:
            def open(self, request, timeout):
                self.request = request
                return Response()
        def build(*args):
            handlers.extend(args)
            return Opener()
        with patch("urllib.request.build_opener", build):
            status, _ = client._default_transport("https://dgx.fixture.invalid/v1/chat/completions", {}, {}, 1)
        self.assertEqual(status, 200)
        self.assertEqual(next(h.proxies for h in handlers if isinstance(h, ProxyHandler)), {})
        redirect = next(h for h in handlers if h.__class__.__name__ == "NoRedirect")
        self.assertIsNone(redirect.redirect_request(None, None, 302, "", {}, "https://other.invalid"))


    def test_each_family_eval_cli_uses_synthetic_cache_or_deterministic_scores(self):
        common_root = self.convert()
        cache = self.root / "synthetic-cache"
        def fixture_transport(url, headers, body, timeout):
            answer = ('{"correct":true,"reason":"synthetic fixture"}'
                      if '"DATA"' in body["messages"][-1]["content"] else "yes")
            return 200, {"choices": [{"message": {"content": answer}}]}
        judges.set_client(judges.DGXClient(transport=fixture_transport, cache_dir=cache))
        for family in evaluate.FAMILIES:
            with self.subTest(family=family):
                qs = evaluate.load_questions(family, common_root / family)
                runs = []
                for q in qs:
                    if family == "locomo":
                        prediction = q["abstention_label"] if q["eval"]["type"] == "abstain_f1" else q["answer"].split(";")[0]
                    elif family == "lme_v2":
                        prediction = "\\boxed{" + q["answer"] + "}"
                    else:
                        prediction = q["answer"] or "Buy a helmet."
                    record = self.invented_record(q, answer_text=prediction)
                    self.assertEqual(evaluate.evaluate_question(q, record)["score"], 1)
                    runs.append(record)
                run = self.dump(f"{family}-run.jsonl", runs, True)
                scored = self.root / f"{family}-scores.json"
                self.cli("bench.eval.evaluate", "--bench", family, "--run", run,
                         "--qs-dir", common_root / family, "--out", scored,
                         "--judge-cache", cache, "--workers", 2)
                result = json.loads(scored.read_text())
                self.assertEqual(result["overall"]["mean_score"], 1)
                self.assertEqual(result["overall"]["n_flagged"], 0)
                if result["judge_stats"]:
                    self.assertEqual(result["judge_stats"]["requests"], 0)
                    self.assertGreater(result["judge_stats"]["cache_hits"], 0)

    def test_eval_cli_refuses_input_path_and_hardlink_overwrites(self):
        qdir = self.root / "qs"
        self.dump("qs/qs_e.jsonl", [{"qid": "q", "answer": "blue", "eval": {"type": "f1"}}], True)
        run = self.dump("input-run.jsonl", [{"qid": "q", "answer_text": "blue"}], True)
        before = run.read_bytes()
        link = self.root / "alias.json"
        os.link(run, link)
        for output in (run, link):
            result = self.cli("bench.eval.evaluate", "--bench", "locomo", "--run", run,
                              "--qs-dir", qdir, "--out", output, ok=False)
            self.assertEqual(result.returncode, 2)
            self.assertEqual(run.read_bytes(), before)

    def test_invalid_endpoint_and_header_config_do_not_echo_secrets(self):
        for url in ("https://user:synthetic-secret@dgx.invalid/v1", "https://dgx.invalid/v1?token=synthetic-secret",
                    "http://dgx.invalid/v1", "https://dgx.invalid/v1#synthetic-secret"):
            config = self.config()
            value = json.loads(config.read_text())
            value["providers"]["dgx-glm53"]["baseUrl"] = url
            config.write_text(json.dumps(value))
            with self.assertRaises(ValueError) as exc:
                judges.DGXClient(models_json=config)
            self.assertNotIn("synthetic-secret", str(exc.exception))


    def invented_record(self, question, **fields):
        # Construct NEW synthetic responses only. Never annotate historical runs.
        declaration = evidence.prepare_question_declaration(question)
        return dict(declaration, run_id="synthetic-run", benchmark=question["family"],
                    variant="base", question_binding_source=evidence.QUESTION_BINDING_SOURCE) | fields

    def bound_case(self, records=None):
        from types import SimpleNamespace
        root = self.root / "binding-common"
        question = {"qid": "q", "env_id": "e", "question": "What color?", "category": "4",
                    "answer": "blue", "eval": {"type": "f1"}}
        self.dump("binding-common/locomo/qs_e.jsonl", [question], True)
        questions, files = evidence.read_questions(root / "locomo")
        records = records or [{"answer_text": "blue"}]
        records = [self.invented_record(questions[0], **row) for row in records]
        run = self.dump("binding-run.jsonl", records, True)
        score_path = self.root / "binding-scores.json"
        self.cli("bench.eval.evaluate", "--run", run, "--bench", "locomo", "--variant", "base",
                 "--dedup", "last", "--qs-dir", root / "locomo", "--out", score_path)
        envs = self.dump("binding-envs.jsonl", [{"env_id": "e", "variant": "base",
            "run_id": "synthetic-run", "benchmark": "locomo",
            "store_snapshot": {"memory": {"m": {"title": "Synthetic", "content": "blue"}}}}], True)
        args = SimpleNamespace(runs=run, scores=score_path, envs=envs, bench="locomo", variant="base",
                               dedup="last", data_common=root, allow_legacy_unverified=False)
        return args, json.loads(score_path.read_text()), questions, files

    def test_choice_association_is_not_uniform_a_and_labels_match_text(self):
        selections, a_scores = set(), []
        for index in range(64):
            question = locomo.build_question({"sample_id": "invented"}, index,
                {"category": 5, "question": "Which unpublished item?", "adversarial_answer": "invented distractor"})
            params = question["eval"]["params"]
            correct = question["abstention_label"]
            selections.add(correct)
            self.assertEqual(question["options"], [f"({label}) {text}" for label, text in question["option_map"].items()])
            self.assertEqual(question["option_map"][correct], "Insufficient information in this record.")
            self.assertEqual(det.score_deterministic("abstain_f1", params, correct, None)["score"], 1)
            other = "b" if correct == "a" else "a"
            self.assertEqual(det.score_deterministic("abstain_f1", params, other, None)["score"], 0)
            a_scores.append(det.score_deterministic("abstain_f1", params, "a", None)["score"])
        self.assertEqual(selections, {"a", "b"})
        self.assertGreater(sum(a_scores), 0)
        self.assertLess(sum(a_scores), len(a_scores))
        self.assertEqual(locomo.CAT_NAMES, {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"})

    def test_raw_cue_stitching_has_explicit_unsupported_boundary(self):
        self.dump("raw-only/locomo_plus.json", [{"time_gap": "one week later"}])
        result = self.cli("bench.converters.run_converters", "--family", "locomo_plus",
                          "--data-root", self.root / "raw-only", "--out-dir", self.root / "out", ok=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("raw duration/stitching source is not distributed", result.stderr)
        self.assertFalse(hasattr(common, "parse_time_gap"))
        self.assertFalse(hasattr(judges, "LP_TEMPLATE_COGNITIVE"))
        self.assertFalse(hasattr(det, "normalize_answer"))

    def test_full_regeneration_replaces_dev_outputs_and_is_documented(self):
        output = self.convert()
        stale = output / "locomo/qs_stale.jsonl"
        stale.write_text("synthetic stale bytes")
        self.cli("bench.converters.run_converters", "--family", "locomo", "--data-root", FIXTURES / "raw",
                 "--out-dir", output, "--full")
        self.assertFalse(stale.exists())
        self.assertEqual(len(evaluate.load_questions("locomo", output / "locomo")), 5)
        text = (ROOT / "bench/converters/run_converters.py").read_text()
        self.assertIn("first regenerates dev outputs", text)

    def test_unversioned_questions_are_not_silently_rescored(self):
        q = {"qid": "old", "answer": "blue", "eval": {"type": "f1"}}
        out = evaluate.evaluate_question(q, {"answer_text": "blue"})
        self.assertEqual((out["flag"], out["score"]), ("eval_error", 0))
        args, scores, questions, files = self.bound_case()
        qfile = args.data_common / "locomo/qs_e.jsonl"
        old = dict(questions[0])
        del old["question_protocol"]
        qfile.write_text(json.dumps(old) + "\n")
        result = self.cli("bench.eval.evaluate", "--run", args.runs, "--bench", "locomo",
                          "--variant", "base", "--qs-dir", qfile.parent, "--out", self.root / "new.json", ok=False)
        self.assertEqual(result.returncode, 2)

    def test_scoring_bindings_cover_questions_evaluator_selection_and_records(self):
        args, scores, questions, files = self.bound_case()
        raw = Path(args.runs).read_bytes()
        selected, records, rows, status = evidence.validate_scores(scores, raw, questions, files,
            family="locomo", variant="base", dedup="last")
        self.assertEqual(status, "verified-local-artifact-consistency")
        binding = scores["scoring_provenance"]
        self.assertEqual(binding["selection"], {"variant": "base", "effective_variant": "base",
                                                "dedup": "last", "qids": ["q"]})
        self.assertEqual(binding["selected_records"]["q"]["sha256"], rows["q"]["run_record_sha256"])
        self.assertEqual(binding["evaluator"]["version"], EVALUATOR_VERSION)
        changed = [dict(questions[0], answer="red")]
        with self.assertRaisesRegex(ValueError, "question-set"):
            evidence.validate_scores(scores, raw, changed, files, family="locomo", variant="base", dedup="last")
        with self.assertRaisesRegex(ValueError, "selection"):
            evidence.validate_scores(scores, raw, questions, files, family="locomo", variant="base", dedup="first")

    def test_attribution_rejects_forged_bindings_and_metadata_even_with_legacy_flag(self):
        args, scores, questions, files = self.bound_case()
        mutations = [
            (["bench"], "locomo_plus"), (["run_sha256"], "0" * 64),
            (["run", "run_id"], "foreign-run"),
            (["scoring_provenance", "question_set_sha256"], "0" * 64),
            (["scoring_provenance", "question_files", 0, "sha256"], "0" * 64),
            (["scoring_provenance", "evaluator", "version"], "other"),
            (["scoring_provenance", "evaluator", "sha256"], "0" * 64),
            (["scoring_provenance", "selection", "variant"], "nomem"),
            (["scoring_provenance", "selection", "dedup"], "first"),
            (["scoring_provenance", "selection", "effective_variant"], "nomem"),
            (["scoring_provenance", "driver_question_binding", "source"], "independent-proof"),
            (["scoring_provenance", "driver_question_binding", "schema"], "obsolete"),
            (["scoring_provenance", "selected_records", "q", "sha256"], "0" * 64),
            (["results", 0, "question_sha256"], "0" * 64),
            (["results", 0, "run_record_sha256"], "0" * 64),
            (["results", 0, "env_id"], "foreign"),
            (["results", 0, "category"], "5"),
            (["results", 0, "eval_type"], "llm_judge"),
            (["results", 0, "scoring_protocol"], "other"),
            (["results", 0, "flag"], "empty_answer"),
            (["results", 0, "score"], True), (["results", 0, "score"], float("nan")),
            (["results", 0, "native_failure", "stop_error"], True),
        ]
        raw = Path(args.runs).read_bytes()
        for path, value in mutations:
            with self.subTest(path=path):
                modified = json.loads(json.dumps(scores))
                node = modified
                for part in path[:-1]:
                    node = node[part]
                node[path[-1]] = value
                with self.assertRaises(ValueError):
                    evidence.validate_scores(modified, raw, questions, files, family="locomo",
                        variant="base", dedup="last", allow_legacy_unverified=True)

    def test_selected_record_hash_prevents_first_last_substitution(self):
        records = [{"qid": "q", "env_id": "e", "variant": "base", "answer_text": "red"},
                   {"qid": "q", "env_id": "e", "variant": "base", "answer_text": "blue"}]
        args, scores, questions, files = self.bound_case(records)
        raw = Path(args.runs).read_bytes()
        scores["scoring_provenance"]["selection"]["dedup"] = "first"
        with self.assertRaisesRegex(ValueError, "selected run-record"):
            evidence.validate_scores(scores, raw, questions, files, family="locomo", variant="base", dedup="first")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            evidence.select_run_records(raw, "base", "error")
        mixed = raw + json.dumps({"qid": "r", "variant": "nomem", "answer_text": "x"}).encode() + b"\n"
        with self.assertRaisesRegex(ValueError, "mixed"):
            evidence.select_run_records(mixed)

    def test_attribution_legacy_optin_has_no_failure_stage_or_overlap_inference(self):
        from bench.analysis import attribute
        args, scores, questions, files = self.bound_case()
        del scores["scoring_provenance"]
        Path(args.scores).write_text(json.dumps(scores))
        with self.assertRaisesRegex(ValueError, "legacy"):
            attribute.attribute(args)
        args.allow_legacy_unverified = True
        result = attribute.attribute(args)
        self.assertEqual(result["binding_status"], "unverified-legacy")
        self.assertEqual(result["per_question"][0]["stage"], "unverified-legacy")
        self.assertNotIn("overlap_ids", result["per_question"][0])
        self.assertNotIn("score", result["per_question"][0])

    def test_flagged_and_native_failure_rows_have_no_overlap_inference(self):
        from bench.analysis import attribute
        native = {"qid": "q", "env_id": "e", "variant": "base", "answer_text": "blue", "stop_reason": "error"}
        args, scores, _, _ = self.bound_case([native])
        result = attribute.attribute(args)
        self.assertEqual(result["per_question"][0]["stage"], "native-failure")
        self.assertNotIn("overlap_ids", result["per_question"][0])
        native["error"] = "synthetic error"
        args, scores, _, _ = self.bound_case([native])
        result = attribute.attribute(args)
        self.assertEqual(result["per_question"][0]["stage"], "score-flagged")
        self.assertNotIn("overlap_ids", result["per_question"][0])

    def test_missing_snapshot_is_not_nomem_and_attribution_stays_noncausal(self):
        from bench.analysis import attribute
        args, scores, _, _ = self.bound_case()
        Path(args.envs).write_text("")
        with self.assertRaisesRegex(ValueError, "missing exact snapshot"):
            attribute.attribute(args)
        args, _, _, _ = self.bound_case()
        result = attribute.attribute(args)
        self.assertEqual(result["per_question"][0]["diagnostic"], "heuristic-overlap-only")
        self.assertIn("not causal", result["interpretation"])
        self.assertIn("not actual presentation", result["binding_claim"])


    def test_obsolete_driver_question_claims_are_rejected_before_scoring(self):
        args, scores, questions, files = self.bound_case()
        row = {"qid": "q", "env_id": "e", "benchmark": "locomo", "variant": "base",
               "run_id": "synthetic-run", "answer_text": "blue", "question": "What color?",
               "question_protocol": "obsolete-protocol", "question_sha256": "0" * 64}
        Path(args.runs).write_text(json.dumps(row) + "\n")
        result = self.cli("bench.eval.evaluate", "--run", args.runs, "--bench", "locomo",
            "--variant", "base", "--qs-dir", args.data_common / "locomo",
            "--out", self.root / "contradictory-scores.json", ok=False)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertFalse((self.root / "contradictory-scores.json").exists())

    def test_implicit_variant_cannot_join_foreign_snapshot_identity(self):
        from bench.analysis import attribute
        args, _, _, _ = self.bound_case()
        self.cli("bench.eval.evaluate", "--run", args.runs, "--bench", "locomo",
            "--qs-dir", args.data_common / "locomo", "--out", args.scores)
        args.variant = None
        foreign = {"env_id": "e", "variant": "nomem", "benchmark": "locomo_plus",
                   "run_id": "foreign-run", "store_snapshot": {"memory": {"m": {"content": "blue"}}}}
        Path(args.envs).write_text(json.dumps(foreign) + "\n")
        with self.assertRaises(ValueError):
            attribute.attribute(args)

    def test_new_preparation_is_gold_free_and_opaque_not_a_presentation_proof(self):
        question = annotate_question({"qid": "invented", "env_id": "e", "question": "Which color?",
            "presented_question": "Which color?\n(a) blue\n(b) red", "answer": "blue",
            "evidence": ["synthetic private evidence"], "category": "4", "eval": {"type": "f1"}}, "locomo")
        declared = evidence.prepare_question_declaration(question)
        self.assertEqual(set(declared), {"qid", "env_id", "question", "question_protocol",
                                        "scoring_protocol", "question_sha256"})
        self.assertEqual(declared["question"], question["presented_question"])
        self.assertEqual(declared["question_sha256"], evidence.canonical_hash(question))
        self.assertNotIn("synthetic private evidence", json.dumps(declared))
        self.assertNotIn("question_binding_source", declared)  # driver, not preparation, labels its echo
        changed_gold = dict(question, answer="red")
        self.assertNotEqual(declared["question_sha256"], evidence.prepare_question_declaration(changed_gold)["question_sha256"])
        record = self.invented_record(question, answer_text="a")
        evidence.validate_driver_question(record, question, "locomo")
        record["question_sha256"] = record["question_sha256"].upper()
        evidence.validate_driver_question(record, question, "locomo")
        record["question"] = question["question"]
        with self.assertRaisesRegex(ValueError, "text/id"):
            evidence.validate_driver_question(record, question, "locomo")

    def test_preparation_cli_requires_explicit_new_output_and_never_reads_predictions(self):
        args, _, questions, _ = self.bound_case()
        output = self.root / "new-driver-questions.jsonl"
        self.cli("bench.prepare_questions", "--bench", "locomo", "--qs-dir", args.data_common / "locomo",
                 "--out", output)
        rows = evidence.jsonl_records(output.read_bytes())
        self.assertEqual(rows, [evidence.prepare_question_declaration(questions[0])])
        self.assertFalse({"answer", "evidence", "eval", "answer_text"} & rows[0].keys())
        before = output.read_bytes()
        result = self.cli("bench.prepare_questions", "--bench", "locomo", "--qs-dir", args.data_common / "locomo",
                          "--out", output, ok=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(output.read_bytes(), before)
        qfile = args.data_common / "locomo/qs_e.jsonl"
        link = self.root / "question-alias.jsonl"
        os.link(qfile, link)
        before_question = qfile.read_bytes()
        result = self.cli("bench.prepare_questions", "--bench", "locomo", "--qs-dir", qfile.parent,
                          "--out", link, ok=False)
        self.assertEqual(result.returncode, 2)
        self.assertEqual(qfile.read_bytes(), before_question)
        self.assertFalse((self.root / "home").exists())

    def test_driver_declarations_require_all_fields_types_limits_and_matching_values(self):
        args, _, questions, _ = self.bound_case()
        question = questions[0]
        record = evidence.jsonl_records(Path(args.runs).read_bytes())[0]
        mutations = {
            "question_protocol": [None, "", " ", 5, "x" * 257, "obsolete"],
            "scoring_protocol": [None, "", [], "x" * 257, "obsolete"],
            "question_sha256": [None, "", 0, "0" * 63, "0" * 65, "g" * 64, "0" * 64],
            "question_binding_source": [None, "", "independently_verified"],
            "question": [None, "", "What color? ", "Other question?"],
            "qid": [None, "foreign"], "run_id": [None, "", " ", 7],
            "benchmark": [None, "", "locomo_plus"], "variant": [None, "", 7],
            "env_id": [None, "", "foreign"],
        }
        for key, values in mutations.items():
            for value in values:
                with self.subTest(key=key, value=value):
                    modified = dict(record, **{key: value})
                    with self.assertRaises(ValueError):
                        evidence.validate_driver_question(modified, question, "locomo")
            with self.subTest(key=key, missing=True):
                modified = dict(record)
                del modified[key]
                with self.assertRaises(ValueError):
                    evidence.validate_driver_question(modified, question, "locomo")

    def test_unbound_predictions_reject_before_any_scoring_or_judge_call(self):
        from types import SimpleNamespace
        args, _, questions, _ = self.bound_case()
        record = evidence.jsonl_records(Path(args.runs).read_bytes())[0]
        for key in ("question_protocol", "scoring_protocol", "question_sha256", "question_binding_source"):
            del record[key]
        Path(args.runs).write_text(json.dumps(record) + "\n")
        out = self.root / "must-not-exist.json"
        eval_args = SimpleNamespace(run=args.runs, bench="locomo", variant="base", dedup="last",
            qs_dir=args.data_common / "locomo", full=False, out=out, limit=0, workers=1, models_json=None)
        with patch.object(evaluate, "evaluate_question", side_effect=AssertionError("scoring started")) as score:
            with self.assertRaises(ValueError):
                evaluate.run(eval_args)
            score.assert_not_called()
        self.assertFalse(out.exists())

    def test_rehashing_inconsistent_driver_claims_cannot_bypass_attribution(self):
        args, scores, questions, files = self.bound_case()
        record = evidence.jsonl_records(Path(args.runs).read_bytes())[0]
        for key, value in (("question_protocol", "obsolete"), ("scoring_protocol", "obsolete"),
                           ("question_sha256", "0" * 64), ("question", "Other visible text?")):
            with self.subTest(key=key):
                modified_record = dict(record, **{key: value})
                raw = (json.dumps(modified_record) + "\n").encode()
                forged = json.loads(json.dumps(scores))
                digest = evidence.canonical_hash(modified_record)
                forged["run_sha256"] = hashlib.sha256(raw).hexdigest()
                forged["scoring_provenance"]["selected_records"]["q"]["sha256"] = digest
                forged["results"][0]["run_record_sha256"] = digest
                with self.assertRaises(ValueError):
                    evidence.validate_scores(forged, raw, questions, files, family="locomo",
                        variant="base", dedup="last", allow_legacy_unverified=True)

    def test_snapshot_join_rejects_each_foreign_or_missing_identity_component(self):
        from bench.analysis import attribute
        args, _, _, _ = self.bound_case()
        snapshot = evidence.jsonl_records(Path(args.envs).read_bytes())[0]
        for key in evidence.IDENTITY_FIELDS:
            for value in (None, "", "foreign"):
                with self.subTest(key=key, value=value):
                    modified = dict(snapshot, **{key: value})
                    Path(args.envs).write_text(json.dumps(modified) + "\n")
                    with self.assertRaises(ValueError):
                        attribute.attribute(args)
            with self.subTest(key=key, missing=True):
                modified = dict(snapshot)
                del modified[key]
                Path(args.envs).write_text(json.dumps(modified) + "\n")
                with self.assertRaises(ValueError):
                    attribute.attribute(args)

    def test_duplicate_exact_snapshot_identity_is_not_resolved_by_run_dedup(self):
        from bench.analysis import attribute
        args, _, _, _ = self.bound_case()
        raw = Path(args.envs).read_bytes()
        Path(args.envs).write_bytes(raw + raw)
        for policy in ("first", "last", "error"):
            with self.subTest(policy=policy):
                args.dedup = policy
                self.cli("bench.eval.evaluate", "--run", args.runs, "--bench", "locomo", "--variant", "base",
                         "--dedup", policy, "--qs-dir", args.data_common / "locomo", "--out", args.scores)
                with self.assertRaisesRegex(ValueError, "ambiguous snapshot"):
                    attribute.attribute(args)

    def test_exact_snapshot_identity_selects_right_sample_when_env_id_is_shared(self):
        from bench.analysis import attribute
        args, _, _, _ = self.bound_case()
        match = evidence.jsonl_records(Path(args.envs).read_bytes())[0]
        foreign = dict(match, run_id="another-run", variant="nomem", benchmark="locomo_plus")
        foreign["store_snapshot"] = {"memory": {"foreign": {"content": "blue"}}}
        Path(args.envs).write_text(json.dumps(foreign) + "\n" + json.dumps(match) + "\n")
        self.cli("bench.eval.evaluate", "--run", args.runs, "--bench", "locomo",
                 "--qs-dir", args.data_common / "locomo", "--out", args.scores)
        args.variant = None
        result = attribute.attribute(args)
        self.assertIsNone(result["variant"])
        self.assertEqual(result["effective_variant"], "base")
        self.assertEqual(result["snapshot_binding_status"], "matched-exact-identities")
        row = result["per_question"][0]
        self.assertEqual(row["overlap_ids"], ["m"])
        self.assertEqual(row["declared_run_identity"]["run_id"], "synthetic-run")
        self.assertEqual(row["snapshot_record_sha256"], evidence.canonical_hash(match))

    def test_implicit_nomem_uses_selected_record_not_absent_cli_argument(self):
        from bench.analysis import attribute
        args, _, _, _ = self.bound_case()
        record = evidence.jsonl_records(Path(args.runs).read_bytes())[0]
        record["variant"] = "nomem"
        Path(args.runs).write_text(json.dumps(record) + "\n")
        self.cli("bench.eval.evaluate", "--run", args.runs, "--bench", "locomo",
                 "--qs-dir", args.data_common / "locomo", "--out", args.scores)
        args.variant = None
        Path(args.envs).write_text("")
        result = attribute.attribute(args)
        self.assertEqual(result["effective_variant"], "nomem")
        self.assertEqual(result["snapshot_binding_status"], "not-applicable")
        self.assertEqual(result["per_question"][0]["diagnostic"], "memory-off-declared")
        self.assertNotIn("overlap_ids", result["per_question"][0])

    def test_missing_snapshot_payload_does_not_invent_empty_memory(self):
        from bench.analysis import attribute
        args, _, _, _ = self.bound_case()
        snapshot = evidence.jsonl_records(Path(args.envs).read_bytes())[0]
        for value in (None, {}, {"memory": None}, {"memory": []}):
            with self.subTest(value=value):
                Path(args.envs).write_text(json.dumps(dict(snapshot, store_snapshot=value)) + "\n")
                with self.assertRaisesRegex(ValueError, "explicit memory"):
                    attribute.attribute(args)
        Path(args.envs).write_text(json.dumps(dict(snapshot, store_snapshot={"memory": {}})) + "\n")
        result = attribute.attribute(args)
        self.assertEqual(result["per_question"][0]["overlap_ids"], [])

    def test_genuinely_unbound_legacy_rows_remain_explicitly_unverified(self):
        from bench.analysis import attribute
        args, scores, _, _ = self.bound_case()
        old_record = {"qid": "q", "answer_text": "blue", "variant": "base"}
        Path(args.runs).write_text(json.dumps(old_record) + "\n")
        scores.pop("scoring_provenance")
        scores.pop("run_sha256")
        Path(args.scores).write_text(json.dumps(scores))
        # Old env-only snapshots have no inferable run identity, even with opt-in.
        Path(args.envs).write_text(json.dumps({"env_id": "e", "store_snapshot": {"memory": {}}}) + "\n")
        with self.assertRaisesRegex(ValueError, "legacy"):
            attribute.attribute(args)
        args.allow_legacy_unverified = True
        result = attribute.attribute(args)
        self.assertEqual(result["binding_status"], "unverified-legacy")
        self.assertEqual(result["snapshot_binding_status"], "unverified-legacy")
        self.assertEqual(result["per_question"][0]["stage"], "unverified-legacy")
        self.assertNotIn("diagnostic", result["per_question"][0])
        self.assertNotIn("overlap_ids", result["per_question"][0])

    def test_old_choice_letters_cannot_be_bound_to_new_option_association(self):
        question = annotate_question(locomo.build_question({"sample_id": "invented-new"}, 0,
            {"category": 5, "question": "Which unpublished color?", "adversarial_answer": "violet"}), "locomo")
        self.dump("new-choice/qs_e.jsonl", [question], True)
        old_record = {"qid": question["qid"], "env_id": question["env_id"], "run_id": "old-synthetic",
                      "benchmark": "locomo", "variant": "base", "question": "Which unpublished color?",
                      "answer_text": "a"}
        run = self.dump("old-choice.jsonl", [old_record], True)
        out = self.root / "must-not-rescore.json"
        result = self.cli("bench.eval.evaluate", "--run", run, "--bench", "locomo", "--variant", "base",
                          "--qs-dir", self.root / "new-choice", "--out", out, ok=False)
        self.assertEqual(result.returncode, 2)
        self.assertFalse(out.exists())
        self.assertEqual(evidence.jsonl_records(run.read_bytes()), [old_record])

    def test_merged_run_ids_bind_each_selected_question_to_its_own_snapshot(self):
        from bench.analysis import attribute
        args, _, questions, _ = self.bound_case()
        second = dict(questions[0], qid="q2", question="What other color?", answer="green")
        self.dump("binding-common/locomo/qs_e.jsonl", [questions[0], second], True)
        questions, _ = evidence.read_questions(args.data_common / "locomo")
        records = [self.invented_record(questions[0], run_id="first-run", answer_text="blue"),
                   self.invented_record(questions[1], run_id="second-run", answer_text="green")]
        self.dump("binding-run.jsonl", records, True)
        snapshots = [{"run_id": record["run_id"], "benchmark": "locomo", "variant": "base", "env_id": "e",
                      "store_snapshot": {"memory": {str(index): {"content": record["answer_text"]}}}}
                     for index, record in enumerate(records)]
        self.dump("binding-envs.jsonl", snapshots, True)
        self.cli("bench.eval.evaluate", "--run", args.runs, "--bench", "locomo", "--variant", "base",
                 "--qs-dir", args.data_common / "locomo", "--out", args.scores)
        result = attribute.attribute(args)
        self.assertEqual([row["overlap_ids"] for row in result["per_question"]], [["0"], ["1"]])
        self.assertEqual([row["declared_run_identity"]["run_id"] for row in result["per_question"]],
                         ["first-run", "second-run"])

    def test_missing_run_rows_keep_flags_without_claiming_a_driver_declaration(self):
        from bench.analysis import attribute
        args, _, _, _ = self.bound_case()
        Path(args.runs).write_text("")
        self.cli("bench.eval.evaluate", "--run", args.runs, "--bench", "locomo", "--variant", "base",
                 "--qs-dir", args.data_common / "locomo", "--out", args.scores)
        scores = json.loads(Path(args.scores).read_text())
        self.assertEqual(scores["results"][0]["flag"], "missing_record")
        self.assertEqual(scores["overall"]["mean_score"], 0)
        self.assertIsNone(scores["scoring_provenance"]["selected_records"]["q"])
        Path(args.envs).write_text("")
        result = attribute.attribute(args)
        self.assertEqual(result["per_question"][0]["stage"], "score-flagged")
        self.assertEqual(result["snapshot_binding_status"], "not-applicable")
        self.assertNotIn("declared_run_identity", result["per_question"][0])

    def test_pinned_licenses_and_independent_module_provenance(self):
        provenance = json.loads((ROOT / "bench/PROVENANCE.json").read_text())
        self.assertEqual(provenance["schema"], "memory-bench-portable-provenance/2")
        for upstream in provenance["upstream"].values():
            path = ROOT / "bench" / upstream["license_file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), upstream["license_sha256"])
        independent = provenance["independent_local_module"]
        text = (ROOT / "bench" / independent["path"]).read_text(encoding="utf-8")
        self.assertEqual(hashlib.sha256(text.encode()).hexdigest(), independent["utf8_lf_sha256"])
        self.assertFalse(independent["official_parity"])
        porter_source = (ROOT / "bench/eval/porter.py").read_text()
        self.assertIn("Modified NLTK", porter_source)
        self.assertIn("Copyright (C) 2001-2026 NLTK Project", porter_source)
        self.assertNotIn("no NLTK code text copied", porter_source)


def load_tests(loader, suite, pattern):
    suite.addTests(loader.loadTestsFromModule(original_scoring_tests))
    return suite


if __name__ == "__main__":
    unittest.main()
