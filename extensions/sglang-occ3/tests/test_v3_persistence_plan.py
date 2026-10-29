"""The wave-3 persistent-candidate bootstrap plan.

Spec section 15: persist_candidate_on_success must not stay implemented but
unused. This pins the real deploy plan artifact for the live campaign
candidate: it parses through the production plan validator, selects the
persistence policy, and passes native admission validation offline. Every
live-observed field is an explicit to-be-filled-at-window placeholder; the
plan cannot run against any real container until those are filled.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deploy.v3 import lifecycle as life
from deploy.v3.native_validator import VALIDATOR_KEY
from occ3_sglang import native_profile

PLAN_PATH = (Path(__file__).resolve().parents[1] / "deploy" / "v3" / "plan-templates"
             / "occ3-icalib-persistent-candidate.json")
ORIGINAL = "0" * 64
CANDIDATE_IMAGE = "sha256:" + "1" * 64


def plan_document():
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def validated_plan():
    return life.validate_plan(plan_document())


def original_doc():
    return {"Id": ORIGINAL, "Image": "sha256:" + "0" * 64, "Name": "/to-be-filled-at-window",
            "Config": {"Image": "sha256:" + "0" * 64, "Hostname": "fixture-host",
                       "Entrypoint": ["python3"],
                       "Cmd": ["-m", "sglang.launch_server", "--api-key", "SYNTHETIC"],
                       "Env": ["A=1"], "WorkingDir": "/w", "User": "", "Labels": {}},
            "HostConfig": {"NetworkMode": "bridge", "Runtime": "runc", "Mounts": [],
                           "Binds": None, "RestartPolicy": {"Name": "always"},
                           "AutoRemove": False, "PortBindings": {}},
            "Mounts": [], "NetworkSettings": {"Networks": {"bridge": {}}},
            "State": {"Running": True, "Restarting": False, "ExitCode": 0,
                      "Status": "running", "StartedAt": "epoch-original"}}


def test_bootstrap_plan_parses_and_selects_candidate_persistence():
    plan = validated_plan()
    assert plan["persistence_policy"] == life.PERSIST_CANDIDATE_ON_SUCCESS
    assert plan["validator"] == VALIDATOR_KEY
    record = plan["preserved_original"]
    assert record["container_id"] == plan["original_id"]
    assert record["image_id"] == plan["original_image_id"]
    # The persistence policy is a plan selection, not a default: the absent
    # field would have selected rollback_on_success.
    assert life.validate_persistence(dict(plan)) == life.PERSIST_CANDIDATE_ON_SUCCESS
    assert life.validate_persistence({k: v for k, v in plan.items()
                                      if k != "persistence_policy"
                                      and k != "preserved_original"}) == life.ROLLBACK_ON_SUCCESS


def test_bootstrap_plan_passes_native_admission_offline(tmp_path):
    plan = validated_plan()
    journal = life.Journal.create(tmp_path / "attempt", plan, original_doc())
    services = life.Services(journal.original(), journal.read()["plan"], journal=journal)
    admission = services.pre_mutation_admission(journal.read())
    assert admission["ok"] is True, admission
    assert admission["method_id"] == native_profile.CALIBRATED_METHOD
    assert admission["admission_scope"] == "offline_inputs_only"
    assert admission["gpu_proof"] == "not_asserted_preflight"


def test_bootstrap_plan_marks_live_fields_to_be_filled():
    plan = plan_document()
    markers = plan["to_be_filled_at_window"]
    assert markers and len(markers) == len(set(markers))
    for path in markers:
        node = plan
        for part in path.split("."):
            assert isinstance(node, dict) and part in node, path
            node = node[part]
    # No fabricated live-verified values: the frozen artifact reference and the
    # pinned harness digests stay placeholder zeros until the window fills them.
    block = plan["native_validator"]
    assert block["admission_template"]["source_profile"]["calibration_digest"] == "0" * 64
    assert block["harness_artifact"]["binary_sha256"] == "0" * 64
    # The committed fingerprint stays internally consistent with the committed
    # (placeholder) identity; the window re-computes it over the real identity.
    import hashlib
    canonical = json.dumps(block["admission_template"]["identity"], sort_keys=True,
                           separators=(",", ":"), ensure_ascii=True)
    assert block["admission_template"]["fingerprint"] == (
        "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest())


def test_bootstrap_plan_refuses_to_be_filled_placeholders_at_admission(tmp_path):
    """A placeholder field that could look live (the plan's declared static
    input digest) is still structurally validated: a non-hex64 fill is
    refused by name, so the window fill cannot weaken admission."""
    from deploy.v3.native_validator import NativeHandoffValidator
    document = plan_document()
    document["native_validator"]["static_input_sha256"] = "not-a-digest"
    plan = life.validate_plan(document)
    journal = life.Journal.create(tmp_path / "attempt", plan, original_doc())
    state = journal.read()
    validator = NativeHandoffValidator(state["plan"], journal, state)
    admission = validator.pre_mutation_admission()
    assert admission["ok"] is False
    assert admission["code"] == "static_input_digest_invalid"
