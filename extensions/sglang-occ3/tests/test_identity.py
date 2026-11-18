"""Identity computation and Rust field-order parity."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from occ3_sglang.identity import IDENTITY_FIELDS, compute_identity, fingerprint

REPO_ROOT = Path(__file__).resolve().parents[3]
RUST_TYPES = (
    REPO_ROOT
    / "crates/pi-coding-agent/src/core/occ/occ3/types.rs"
)
PROFILE_INPUT = (
    Path(__file__).resolve().parents[1]
    / "deploy/native-build-inputs/native-profile-input.json"
)

# The declared-launch identity pinned in the packaged build input. The
# latent_method label is the legacy bridge label: the optimus-agent client
# (activation.rs mint_ready) replaces it with the runtime-attested method id
# before hashing, and that substitution must bind the pinned fingerprint.
DECLARED_IDENTITY = {
    "model_id": "glm-5.3",
    "architecture": "GlmMoeDsaForCausalLM",
    "checkpoint_revision": "aca966e4e02791568aa6a4ced368624b3d897f42",
    "weight_fingerprint": "metadata-manifest-sha256:493a065ae286f3ca2077d36e8d0fd0bb931d7443d6dab309e7e2cf288c32d23e",
    "tokenizer_fingerprint": "sha256:08d2801689875e6819f26919b3e07ecf42693dd58e0527be477de286fb8ef532",
    "chat_template_fingerprint": "sha256:3740abcea51c45830cb3ca562084ad5fb2ef53589376f73332e9886f93ade41c",
    "quantisation": "fp8",
    "tensor_geometry": "layers78:hidden6144:prenorm-layer77",
    "kv_layout": "declared-launch:tp8:kv-fp8_e4m3:continuous-prefix",
    "latent_method": "kv-radix-channel-v1",
    "runtime": "sglang:0.5.21:e00930c5489053f26d86b179cee0d087f846acbb",
}
PINNED_PERMIT_FINGERPRINT = (
    "sha256:efef608a8f55388d16e16af8ced7b6936f95c018ee098ae6665c69fcb8f90da8"
)


def _declared_block():
    value = json.loads(PROFILE_INPUT.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value.get("declared_identity")


def test_identity_has_eleven_fields_in_contract_order():
    assert len(IDENTITY_FIELDS) == 11


def test_identity_field_order_matches_rust_bridge():
    """The fingerprint string must be comparable across languages."""
    source = RUST_TYPES.read_text(encoding="utf-8")
    start = source.find("pub struct Occ3ModelIdentity")
    end = source.find("impl Occ3ModelIdentity", start)
    struct = source[start:end]
    fields = [
        line.split(":")[0].replace("pub ", "").strip()
        for line in struct.splitlines()
        if line.strip().startswith("pub ") and ":" in line
    ]
    assert fields, "failed to parse Occ3ModelIdentity fields"
    assert tuple(fields) == IDENTITY_FIELDS


def test_identity_computed_from_stub_tm(stub_tm):
    identity = compute_identity(stub_tm)
    assert set(identity) == set(IDENTITY_FIELDS)
    assert identity["architecture"] == "GlmMoeDsaForCausalLM"
    assert identity["model_id"] == "glm-5.3"
    assert identity["checkpoint_revision"] == "aca966e4e02791568aa6a4ced368624b3d897f42"
    assert identity["quantisation"] == "fp8-e4m3"
    assert "hidden_size=6144" in identity["tensor_geometry"]
    assert "tp_size=8" in identity["kv_layout"]
    assert "kv_cache_dtype=fp8_e4m3" in identity["kv_layout"]
    assert identity["latent_method"] == "kv-radix-channel-v1"
    assert identity["runtime"] == "sglang/0.5.21-test"


def test_fingerprint_is_stable_and_pipe_joined(stub_tm):
    identity = compute_identity(stub_tm)
    fp = fingerprint(identity)
    assert fp == "|".join(identity[f] for f in IDENTITY_FIELDS)
    assert fingerprint(compute_identity(stub_tm)) == fp


def test_fingerprint_changes_when_topology_changes(stub_tm):
    before = fingerprint(compute_identity(stub_tm))
    stub_tm.server_args.tp_size = 4
    after = fingerprint(compute_identity(stub_tm))
    assert before != after


def test_identity_never_crashes_on_minimal_tm():
    class Bare:
        pass

    identity = compute_identity(Bare())
    assert set(identity) == set(IDENTITY_FIELDS)
    assert all(isinstance(v, str) for v in identity.values())
    # Unknown runtime yields the fallback marker, not an exception.
    assert identity["runtime"] in ("sglang/unknown",) or identity["runtime"].startswith("sglang/")


def test_declared_identity_block_matches_the_launch_declaration():
    """The pinned block in the build input is exactly the declared launch."""
    assert _declared_block() == DECLARED_IDENTITY
    # The route serves the legacy label; only the client substitutes the id.
    assert DECLARED_IDENTITY["latent_method"] == "kv-radix-channel-v1"


def test_declared_identity_field_names_match_the_identity_contract():
    """The validator's block fieldset is pinned to the contract order."""
    from occ3_sglang.native_profile import _DECLARED_IDENTITY_FIELDS

    assert _DECLARED_IDENTITY_FIELDS == set(IDENTITY_FIELDS)


def test_declared_identity_binding_recipe_pins_the_calibrated_fingerprint():
    """The client's permit binding, reproduced offline, must hold exactly.

    activation.rs mint_ready replaces latent_method with the runtime-attested
    method id, then digests the eleven fields as canonical JSON (sorted keys,
    compact separators, ASCII). That digest must equal the pinned calibrated
    model fingerprint, so the served declared fields bind the permit.
    """
    from occ3_sglang.native_profile import CALIBRATED_METHOD

    fields = dict(_declared_block())
    fields["latent_method"] = CALIBRATED_METHOD
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=True)
    digest = "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert digest == PINNED_PERMIT_FINGERPRINT
