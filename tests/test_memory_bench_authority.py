"""Offline regression coverage for the applied LongMemEval-V2 converter.

All inputs are invented here and written beneath a temporary directory. Tests
import the checkout's real converter; no proposal copy, dataset, Rust mirror,
provider, or model is used. Native writer/legacy-policy tests remain in Rust.
"""
from __future__ import annotations

import copy
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from bench import protocol
from bench.converters import common, lme_v2

CAPTURE_PROTOCOL = "optimus-memory-capture/temporal-authority/1.0.0"
QUESTION_PROTOCOL = "optimus-lme-v2-adapter/2.0.0"
SCORING_PROTOCOL = "optimus-lme-v2-dgx-port/2.0.0"


class MemoryBenchAuthorityTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="memory-bench-authority-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.raw = self.root / "invented raw"
        self.raw.mkdir()
        self.output = self.root / "common"
        for module, name, value in (
            (common, "DATA_ROOT", str(self.raw)),
            (common, "COMMON_ROOT", str(self.output)),
            (lme_v2, "AXTREE_CAP", 8000),
        ):
            patcher = patch.object(module, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.trajectories = [
            {
                "id": "fixture-web-z",
                "goal": "Open  an invented panel.\nKeep the final newline.\n",
                "outcome": "Synthetic completion is not user confirmation.",
                "start_url": "https://fixture.invalid/start",
                "environment": "invented-browser",
                "states": [
                    {
                        "state_index": 7,
                        "url": "https://fixture.invalid/panel?label=λ&view=two",
                        "accessibility_tree": "heading 'Synthetic lantern'\n\n  row 'Amber  Ω'\n",
                        "action": "click('Save')\n  wait(3)\n",
                        "thought": "Keep  both  spaces.\nNo user has confirmed this.\n",
                        "screenshot": "SYNTHETIC_SCREENSHOT_NOT_INGESTED",
                    },
                    {
                        "state_index": 2,
                        "url": "https://fixture.invalid/review",
                        "accessibility_tree": "button 'Review'\nlabel 'invented'",
                        "action": None,
                        "thought": "Review the invented panel.\n",
                    },
                    {
                        "state_index": 11,
                        "url": None,
                        "accessibility_tree": None,
                        "action": "open('invented-summary')",
                        "thought": "",
                    },
                ],
            },
            {
                "id": "fixture-excluded",
                "goal": "UNSELECTED_TRAJECTORY_SENTINEL",
                "outcome": "unused",
                "start_url": "https://fixture.invalid/excluded",
                "states": [],
            },
            {
                "id": "fixture-enterprise-m",
                "goal": "Inspect an invented inventory.",
                "outcome": "Synthetic outcome.",
                "start_url": "https://fixture.invalid/inventory",
                "environment": "invented-enterprise",
                "states": [
                    {
                        "state_index": 9,
                        "url": "https://fixture.invalid/items",
                        "accessibility_tree": "row 'Invented copper lamp'",
                        "action": None,
                        "thought": None,
                    },
                    {
                        "state_index": 1,
                        "url": "",
                        "accessibility_tree": "λ雪🙂éΩx",
                        "action": "select('invented-item')",
                        "thought": "The observed row is synthetic.",
                    },
                ],
            },
            {
                "id": "fixture-web-a",
                "goal": "Inspect an empty invented state.",
                "outcome": "Synthetic empty result.",
                "start_url": "",
                "environment": "invented-browser",
                "states": [{"state_index": 5}],
            },
        ]
        self.questions = [
            {
                "id": "fixture-q-web-z", "domain": "web",
                "question": "Invented prompt: name the synthetic color.",
                "answer": "INVENTED_ANSWER_AMBER", "question_type": "synthetic-fields",
                "eval_function": "norm_phrase_set_match|case_sensitive=false|separator=;|limit=3",
                "environment": "invented-browser", "image": None,
            },
            {
                "id": "fixture-q-enterprise-z", "domain": "enterprise",
                "question": "Invented prompt: choose the synthetic item.",
                "answer": "B", "question_type": "synthetic-choice",
                "eval_function": "mc_choice_match|split_multiple_boxes=true",
                "environment": "invented-enterprise",
            },
            {
                "id": "fixture-q-web-a", "domain": "web",
                "question": "Invented prompt: describe the synthetic gotcha.",
                "answer": "INVENTED_ANSWER_GOTCHA", "question_type": "errors-gotchas",
                "eval_function": "llm_gotchas_checker",
                "environment": "invented-browser", "image": "invented-only.png",
            },
            {
                "id": "fixture-q-enterprise-a", "domain": "enterprise",
                "question": "Invented prompt: order the synthetic labels.",
                "answer": "INVENTED_FIRST; INVENTED_SECOND",
                "question_type": "synthetic-order",
                "eval_function": "norm_phrase_set_match_ordered|split_multiple_boxes=false",
                "environment": "invented-enterprise",
            },
        ]
        self.haystack = {
            question["id"]: (["fixture-web-a", "fixture-web-z"]
                             if question["domain"] == "web"
                             else ["fixture-enterprise-m"])
            for question in self.questions
        }
        self._write_raw()

    def _write_raw(self):
        for name, rows in (("trajectories.jsonl", self.trajectories),
                           ("lme_v2_questions.jsonl", self.questions)):
            (self.raw / name).write_text(
                "\n" + "\n\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
                encoding="utf-8",
            )
        (self.raw / "lme_v2_small_haystack.json").write_text(
            json.dumps(self.haystack), encoding="utf-8")

    def _domain(self, domain="web"):
        ids = (["fixture-web-a", "fixture-web-z"] if domain == "web"
               else ["fixture-enterprise-m"])
        return lme_v2.build_domain_env(domain, ids)

    def _build(self, dev_only=False):
        manifest = lme_v2.build(dev_only=dev_only)
        output = self.output / "lme_v2"
        envs = {domain: json.loads((output / f"env_{domain}.json").read_text(encoding="utf-8"))
                for domain in ("web", "enterprise")}
        return manifest, envs, output

    def _read_rows(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def _expected_questions(self):
        # Fixed expected rows, independent of annotate_question/build_question.
        return [
            {
                "qid": "fixture-q-web-z", "env_id": "web",
                "question": "Invented prompt: name the synthetic color.",
                "answer": "INVENTED_ANSWER_AMBER", "evidence": [],
                "category": "synthetic-fields",
                "eval": {"type": "phrase_set", "params": {
                    "case_sensitive": False, "separator": ";", "limit": "3"},
                    "function": "norm_phrase_set_match", "protocol": SCORING_PROTOCOL},
                "domain": "web", "environment": "invented-browser", "image": None,
                "raw_eval_function": "norm_phrase_set_match|case_sensitive=false|separator=;|limit=3",
                "family": "lme_v2", "question_protocol": QUESTION_PROTOCOL,
            },
            {
                "qid": "fixture-q-enterprise-z", "env_id": "enterprise",
                "question": "Invented prompt: choose the synthetic item.",
                "answer": "B", "evidence": [], "category": "synthetic-choice",
                "eval": {"type": "mc_choice", "params": {"split_multiple_boxes": True},
                         "function": "mc_choice_match", "protocol": SCORING_PROTOCOL},
                "domain": "enterprise", "environment": "invented-enterprise", "image": None,
                "raw_eval_function": "mc_choice_match|split_multiple_boxes=true",
                "family": "lme_v2", "question_protocol": QUESTION_PROTOCOL,
            },
            {
                "qid": "fixture-q-web-a", "env_id": "web",
                "question": "Invented prompt: describe the synthetic gotcha.",
                "answer": "INVENTED_ANSWER_GOTCHA", "evidence": [], "category": "errors-gotchas",
                "eval": {"type": "llm_gotchas", "params": {},
                         "function": "llm_gotchas_checker", "protocol": SCORING_PROTOCOL},
                "domain": "web", "environment": "invented-browser", "image": "invented-only.png",
                "raw_eval_function": "llm_gotchas_checker",
                "family": "lme_v2", "question_protocol": QUESTION_PROTOCOL,
            },
            {
                "qid": "fixture-q-enterprise-a", "env_id": "enterprise",
                "question": "Invented prompt: order the synthetic labels.",
                "answer": "INVENTED_FIRST; INVENTED_SECOND", "evidence": [],
                "category": "synthetic-order",
                "eval": {"type": "phrase_set_ordered", "params": {"split_multiple_boxes": False},
                         "function": "norm_phrase_set_match_ordered", "protocol": SCORING_PROTOCOL},
                "domain": "enterprise", "environment": "invented-enterprise", "image": None,
                "raw_eval_function": "norm_phrase_set_match_ordered|split_multiple_boxes=false",
                "family": "lme_v2", "question_protocol": QUESTION_PROTOCOL,
            },
        ]

    def test_imports_resolve_to_the_applied_checkout(self):
        for module, relative in ((lme_v2, "bench/converters/lme_v2.py"),
                                 (common, "bench/converters/common.py"),
                                 (protocol, "bench/protocol.py")):
            with self.subTest(module=module.__name__):
                self.assertEqual(Path(module.__file__).resolve(), (ROOT / relative).resolve())

    def test_output_roles_and_authorities_never_invent_user(self):
        _, envs, _ = self._build()
        observed = set()
        for env in envs.values():
            for session in env["sessions"]:
                for event in session["events"]:
                    observed.add((event["role"], event["authority"]))
                    self.assertNotEqual(event["role"], "user")
        self.assertEqual(observed, {
            ("assistant", "dataset_task_metadata"),
            ("toolResult", "browser_observation"),
            ("assistant", "agent_decision"),
        })

    def test_header_preserves_text_and_stays_assistant_metadata(self):
        env, _ = self._domain()
        self.assertEqual(env["sessions"][0]["events"][0], {
            "role": "assistant", "ext_id": "fixture-web-z:header", "ts": None,
            "authority": "dataset_task_metadata",
            "text": "Goal: Open  an invented panel.\nKeep the final newline.\n"
                    "\nOutcome: Synthetic completion is not user confirmation."
                    "\nStart: https://fixture.invalid/start",
        })
        self.assertEqual(env["sessions"][0]["traj_meta"], {
            "outcome": "Synthetic completion is not user confirmation.",
            "environment": "invented-browser",
        })

    def test_trajectory_file_order_not_haystack_order_is_preserved(self):
        env, _ = self._domain()
        self.assertEqual([session["session_id"] for session in env["sessions"]],
                         ["fixture-web-z", "fixture-web-a"])
        self.assertNotIn("UNSELECTED_TRAJECTORY_SENTINEL", json.dumps(env))

    def test_states_keep_source_list_order_observation_then_decision(self):
        env, _ = self._domain()
        self.assertEqual([(event["role"], event["ext_id"])
                          for event in env["sessions"][0]["events"]], [
            ("assistant", "fixture-web-z:header"),
            ("toolResult", "fixture-web-z:s7:obs"), ("assistant", "fixture-web-z:s7:act"),
            ("toolResult", "fixture-web-z:s2:obs"), ("assistant", "fixture-web-z:s2:act"),
            ("toolResult", "fixture-web-z:s11:obs"), ("assistant", "fixture-web-z:s11:act"),
        ])

    def test_state_text_preserves_whitespace_unicode_and_field_order(self):
        env, _ = self._domain()
        self.assertEqual([event["text"] for event in env["sessions"][0]["events"][1:]], [
            "URL: https://fixture.invalid/panel?label=λ&view=two\nObservation:\n"
            "heading 'Synthetic lantern'\n\n  row 'Amber  Ω'\n",
            "Action: click('Save')\n  wait(3)\n\nThought: Keep  both  spaces.\n"
            "No user has confirmed this.\n",
            "URL: https://fixture.invalid/review\nObservation:\nbutton 'Review'\nlabel 'invented'",
            "Thought: Review the invented panel.\n",
            "URL: \nObservation:\n",
            "Action: open('invented-summary')",
        ])
        enterprise, _ = self._domain("enterprise")
        self.assertEqual([event["text"] for event in enterprise["sessions"][0]["events"][1:]], [
            "URL: https://fixture.invalid/items\nObservation:\nrow 'Invented copper lamp'",
            "URL: \nObservation:\nλ雪🙂éΩx",
            "Action: select('invented-item')\nThought: The observed row is synthetic.",
        ])

    def test_missing_or_empty_decisions_emit_only_observations(self):
        for fields in ({}, {"action": None, "thought": None}, {"action": "", "thought": ""}):
            with self.subTest(fields=fields):
                self.trajectories[-1]["states"] = [{"state_index": 5, **fields}]
                self._write_raw()
                env, stats = self._domain()
                self.assertEqual(env["sessions"][1]["events"][1:], [{
                    "role": "toolResult", "text": "URL: \nObservation:\n",
                    "ext_id": "fixture-web-a:s5:obs", "ts": None,
                    "authority": "browser_observation",
                    "tool_result": {"toolCallId": "fixture-web-a:s5:obs",
                                    "toolName": "lme_v2_observation", "isError": False},
                }])
                self.assertEqual(stats["states_without_decision"], 1)

    def test_tool_descriptor_has_exact_native_keys_and_values(self):
        _, envs, _ = self._build()
        tools = 0
        for env in envs.values():
            for session in env["sessions"]:
                for event in session["events"]:
                    if event["role"] != "toolResult":
                        self.assertNotIn("tool_result", event)
                        continue
                    tools += 1
                    self.assertEqual(event["tool_result"], {
                        "toolCallId": event["ext_id"],
                        "toolName": "lme_v2_observation", "isError": False,
                    })
                    self.assertIs(type(event["tool_result"]["toolCallId"]), str)
                    self.assertTrue(event["tool_result"]["toolCallId"])
                    self.assertIs(type(event["tool_result"]["toolName"]), str)
                    self.assertIs(event["tool_result"]["isError"], False)
        self.assertEqual(tools, 6)

    def test_ext_ids_are_unique_and_stable_across_rebuilds(self):
        _, first, _ = self._build()
        _, second, _ = self._build()
        ids = [event["ext_id"] for env in first.values()
               for session in env["sessions"] for event in session["events"]]
        self.assertEqual(len(ids), 13)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(first, second)

    def test_no_session_or_event_timestamp_is_invented(self):
        _, envs, _ = self._build()
        for env in envs.values():
            for session in env["sessions"]:
                self.assertIn("ts", session)
                self.assertIsNone(session["ts"])
                self.assertNotIn("timestamp", session)
                for event in session["events"]:
                    self.assertIn("ts", event)
                    self.assertIsNone(event["ts"])
                    self.assertNotIn("timestamp", event)

    def test_only_trajectory_text_is_ingested_no_questions_or_screenshots(self):
        _, envs, _ = self._build()
        for env in envs.values():
            self.assertEqual(env["ground_truth_refs"], [])
            text = "\n".join(event["text"] for session in env["sessions"] for event in session["events"])
            for sentinel in ("Invented prompt:", "INVENTED_ANSWER_", "INVENTED_FIRST",
                             "SYNTHETIC_SCREENSHOT_NOT_INGESTED", "invented-only.png"):
                self.assertNotIn(sentinel, text)

    def test_default_cap_and_environment_override_use_the_real_module(self):
        with patch.dict(os.environ):
            os.environ.pop("LME_V2_AXTREE_CAP", None)
            importlib.reload(lme_v2)
            self.assertEqual(lme_v2.AXTREE_CAP, 8000)
            self.assertEqual(lme_v2.cap_accessibility_tree("x" * 8000), ("x" * 8000, ""))
            self.assertEqual(lme_v2.cap_accessibility_tree("x" * 8001), (
                "x" * 8000, "\n[accessibility tree truncated: first 8000 of 8001 chars]"))
            os.environ["LME_V2_AXTREE_CAP"] = "6"
            importlib.reload(lme_v2)
            self.assertEqual(lme_v2.AXTREE_CAP, 6)
            self.assertEqual(lme_v2.cap_accessibility_tree("ABCDEFG"), (
                "ABCDEF", "\n[accessibility tree truncated: first 6 of 7 chars]"))

    def test_cap_boundaries_count_characters_not_encoded_bytes(self):
        with patch.object(lme_v2, "AXTREE_CAP", 6):
            for tree in ("", "12345", "λ雪🙂éΩx"):
                with self.subTest(length=len(tree)):
                    self.assertEqual(lme_v2.cap_accessibility_tree(tree), (tree, ""))
            self.assertEqual(lme_v2.cap_accessibility_tree("λ雪🙂éΩxy"), (
                "λ雪🙂éΩx", "\n[accessibility tree truncated: first 6 of 7 chars]"))

    def test_cap_markers_and_counts_exclude_header_url_and_decisions(self):
        trees = ["", "12345", "λ雪🙂éΩx", "ABCDEFG", None]
        self.trajectories[-1]["states"] = [
            {"state_index": index, "url": "https://fixture.invalid/long-url",
             "accessibility_tree": tree, "action": "long synthetic action",
             "thought": "long synthetic thought"}
            for index, tree in enumerate(trees)
        ] + [{"state_index": 5}]
        self._write_raw()
        with patch.object(lme_v2, "AXTREE_CAP", 6):
            env, stats = lme_v2.build_domain_env("web", ["fixture-web-a"])
        self.assertEqual(stats, {
            "states_total": 6, "states_truncated": 1,
            "axtree_chars_raw": 18, "axtree_chars_capped": 17,
            "events_header": 1, "events_observation": 6, "events_decision": 5,
            "states_without_decision": 1,
        })
        self.assertEqual({key: env["notes"][key] for key in stats}, stats)
        self.assertEqual(env["notes"]["axtree_cap_chars"], 6)
        events = env["sessions"][0]["events"]
        self.assertEqual(events[0]["text"],
                         "Goal: Inspect an empty invented state.\nOutcome: Synthetic empty result.\nStart: ")
        observations = [event for event in events if event["role"] == "toolResult"]
        self.assertEqual([event["text"] for event in observations], [
            "URL: https://fixture.invalid/long-url\nObservation:\n" + body
            for body in ("", "12345", "λ雪🙂éΩx",
                         "ABCDEF\n[accessibility tree truncated: first 6 of 7 chars]", "")
        ] + ["URL: \nObservation:\n"])
        decisions = [event for event in events if event["authority"] == "agent_decision"]
        self.assertEqual([event["text"] for event in decisions], [
            "Action: long synthetic action\nThought: long synthetic thought"] * 5)
        self.assertEqual(sum(event["text"].count("[accessibility tree truncated:")
                             for event in events), 1)

    def test_zero_cap_keeps_observation_and_marker_without_tree_text(self):
        self.trajectories[-1]["states"] = [{
            "state_index": 5, "url": "https://fixture.invalid/zero",
            "accessibility_tree": "λ雪🙂", "action": "keep action",
        }]
        self._write_raw()
        with patch.object(lme_v2, "AXTREE_CAP", 0):
            env, stats = lme_v2.build_domain_env("web", ["fixture-web-a"])
        self.assertEqual(env["sessions"][0]["events"][1]["text"],
                         "URL: https://fixture.invalid/zero\nObservation:\n"
                         "\n[accessibility tree truncated: first 0 of 3 chars]")
        self.assertEqual(env["sessions"][0]["events"][2]["text"], "Action: keep action")
        self.assertEqual((stats["states_truncated"], stats["axtree_chars_raw"],
                          stats["axtree_chars_capped"]), (1, 3, 0))

    def test_manifest_counts_and_character_totals_match_actual_events(self):
        manifest, envs, _ = self._build()
        expected = {"web": (2, 4, 3, 1), "enterprise": (1, 2, 1, 1)}
        total_chars = 0
        for meta in manifest["envs"]:
            env = envs[meta["env_id"]]
            headers, observations, decisions, no_decisions = expected[meta["env_id"]]
            for key, count in (("events_header", headers), ("events_observation", observations),
                               ("events_decision", decisions)):
                self.assertEqual(env["notes"][key], count)
                self.assertEqual(meta[key], count)
            self.assertEqual(env["notes"]["states_total"], observations)
            self.assertEqual(env["notes"]["states_without_decision"], no_decisions)
            self.assertEqual(meta["n_sessions"], headers)
            self.assertEqual(meta["n_events"], headers + observations + decisions)
            chars = sum(len(event["text"]) for session in env["sessions"] for event in session["events"])
            self.assertEqual(meta["chars"], chars)
            self.assertEqual(meta["est_tokens"], chars // 4)
            total_chars += chars
        self.assertEqual(manifest["n_envs"], 2)
        self.assertEqual(manifest["n_questions"], 4)
        self.assertEqual(manifest["total_env_chars"], total_chars)
        self.assertEqual(manifest["estimated_ingest_tokens"], total_chars // 4)

    def test_capped_build_propagates_per_domain_tree_counts_to_manifest(self):
        with patch.object(lme_v2, "AXTREE_CAP", 6):
            manifest, envs, _ = self._build()
        selected = {"web": [self.trajectories[0], self.trajectories[-1]],
                    "enterprise": [self.trajectories[2]]}
        for meta in manifest["envs"]:
            domain = meta["env_id"]
            trees = [state.get("accessibility_tree") or "" for trajectory in selected[domain]
                     for state in trajectory["states"]]
            expected = {"states_truncated": sum(len(tree) > 6 for tree in trees),
                        "axtree_chars_raw": sum(map(len, trees)),
                        "axtree_chars_capped": sum(min(len(tree), 6) for tree in trees)}
            for key, count in expected.items():
                self.assertEqual(meta[key], count)
                self.assertEqual(envs[domain]["notes"][key], count)
            self.assertEqual(envs[domain]["notes"]["axtree_cap_chars"], 6)

    def test_question_rows_remain_exactly_the_existing_question_contract(self):
        _, _, output = self._build()
        expected = self._expected_questions()
        for domain in ("web", "enterprise"):
            rows = [row for row in expected if row["env_id"] == domain]
            self.assertEqual(self._read_rows(output / f"qs_{domain}.jsonl"),
                             sorted(rows, key=lambda row: row["qid"]))
            self.assertEqual(self._read_rows(output / "full" / f"qs_{domain}.jsonl"), rows)
        full = json.loads((output / "full" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(full["questions"], expected)
        self.assertEqual(full["n_questions"], 4)

    def test_changing_capture_cap_leaves_question_files_byte_identical(self):
        _, before, output = self._build()
        names = ("qs_web.jsonl", "qs_enterprise.jsonl", "full/qs_web.jsonl",
                 "full/qs_enterprise.jsonl", "full/manifest.json")
        original = {name: (output / name).read_bytes() for name in names}
        with patch.object(lme_v2, "AXTREE_CAP", 6):
            _, after, _ = self._build()
        self.assertNotEqual(before, after)
        self.assertEqual({name: (output / name).read_bytes() for name in names}, original)

    def test_capture_revision_is_separate_from_unchanged_question_protocol(self):
        manifest, envs, output = self._build()
        full = json.loads((output / "full" / "manifest.json").read_text(encoding="utf-8"))
        for artifact in (manifest, full, *envs.values()):
            self.assertEqual(artifact["family"], "lme_v2")
            self.assertIs(type(artifact["authority_revision"]), int)
            self.assertEqual(artifact["authority_revision"], 1)
            self.assertEqual(artifact["capture_protocol"], CAPTURE_PROTOCOL)
            self.assertEqual(artifact["question_protocol"], QUESTION_PROTOCOL)
        for row in full["questions"]:
            self.assertNotIn("capture_protocol", row)
            self.assertNotIn("authority_revision", row)
            self.assertEqual(row["eval"]["protocol"], SCORING_PROTOCOL)
            protocol.validate_question_protocol(row, "lme_v2")

    def test_authority_policy_documents_no_user_and_absent_event_times(self):
        env, _ = self._domain()
        self.assertEqual(env["authority_policy"]["revision"], 1)
        self.assertEqual(env["authority_policy"]["user"],
                         "none: the dataset records no user turns; no user origin is invented")
        self.assertEqual(env["authority_policy"]["missing_ts"],
                         "absent: dataset has no event times; no synthetic fallback")
        self.assertTrue(env["authority_policy"]["header"].startswith("assistant:"))
        self.assertTrue(env["authority_policy"]["observation"].startswith("toolResult:"))
        self.assertTrue(env["authority_policy"]["decision"].startswith("assistant:"))

    def test_provenance_and_protocol_keep_separate_capture_identity(self):
        provenance = json.loads((ROOT / "bench/PROVENANCE.json").read_text(encoding="utf-8"))
        self.assertEqual(protocol.CAPTURE_PROTOCOL_VERSION, CAPTURE_PROTOCOL)
        self.assertEqual(provenance["capture_protocol"], CAPTURE_PROTOCOL)
        self.assertEqual(protocol.QUESTION_PROTOCOLS["lme_v2"], QUESTION_PROTOCOL)
        self.assertEqual(protocol.SCORING_PROTOCOLS["lme_v2"], SCORING_PROTOCOL)
        self.assertEqual(protocol.EVALUATOR_VERSION, "optimus-memory-evaluation/2.0.0")
        self.assertEqual(provenance["evaluator_version"], protocol.EVALUATOR_VERSION)
        self.assertEqual(protocol.LOCAL_SCORING_VERSION, "optimus-local-scoring/1.0.0")
        self.assertEqual(provenance["local_scoring_version"], protocol.LOCAL_SCORING_VERSION)

    def test_dev_only_still_writes_full_planning_manifest_not_full_question_files(self):
        manifest, _, output = self._build(dev_only=True)
        self.assertEqual(manifest["n_questions"], 4)
        full = json.loads((output / "full" / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(full["questions"], self._expected_questions())
        self.assertEqual(full["capture_protocol"], CAPTURE_PROTOCOL)
        self.assertEqual(full["authority_revision"], 1)
        self.assertFalse((output / "full" / "qs_web.jsonl").exists())
        self.assertFalse((output / "full" / "qs_enterprise.jsonl").exists())

    def test_build_question_does_not_mutate_source_row(self):
        source = copy.deepcopy(self.questions[0])
        self.assertEqual(lme_v2.build_question(self.questions[0]), self._expected_questions()[0])
        self.assertEqual(self.questions[0], source)

    def test_missing_selected_trajectory_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, "haystack mismatch for web: missing 1, extra 0"):
            lme_v2.build_domain_env("web", ["fixture-web-z", "fixture-not-present"])

    def test_nonshared_haystack_within_domain_is_rejected(self):
        self.haystack["fixture-q-web-a"] = ["fixture-web-z"]
        self._write_raw()
        with self.assertRaisesRegex(RuntimeError, "haystack not shared within web"):
            lme_v2.build()

    def test_missing_required_header_fields_are_rejected(self):
        original = copy.deepcopy(self.trajectories[0])
        for key in ("goal", "outcome", "start_url"):
            with self.subTest(key=key):
                self.trajectories[0] = copy.deepcopy(original)
                del self.trajectories[0][key]
                self._write_raw()
                with self.assertRaises(KeyError) as raised:
                    self._domain()
                self.assertEqual(raised.exception.args, (key,))

    def test_missing_required_state_index_is_rejected(self):
        del self.trajectories[0]["states"][0]["state_index"]
        self._write_raw()
        with self.assertRaises(KeyError) as raised:
            self._domain()
        self.assertEqual(raised.exception.args, ("state_index",))

    def test_nontext_accessibility_tree_is_not_silently_stringified(self):
        self.trajectories[0]["states"][0]["accessibility_tree"] = {"nodes": ["invented"]}
        self._write_raw()
        with self.assertRaises(TypeError):
            self._domain()

    def test_unsupported_question_evaluator_is_rejected(self):
        question = dict(self.questions[0], eval_function="unsupported_fixture_evaluator")
        with self.assertRaisesRegex(ValueError, "unknown eval_function: unsupported_fixture_evaluator"):
            lme_v2.build_question(question)

    def test_malformed_trajectory_json_is_rejected(self):
        (self.raw / "trajectories.jsonl").write_text('{"id": "invented",', encoding="utf-8")
        with self.assertRaises(json.JSONDecodeError):
            self._domain()


if __name__ == "__main__":
    unittest.main()
