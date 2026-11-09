"""Offline staging inventory guards; no deployment or service operations."""
import importlib.util
from pathlib import Path


def test_stage_plugin_inventory_pin_matches_current_sources():
    root = Path(__file__).resolve().parents[1]
    path = root / "deploy/v3/live-instrumentation/stage_build.py"
    spec = importlib.util.spec_from_file_location("occ3_stage_build_inventory", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    names, ignored, excluded = module.plugin_files()
    assert module.PLUGIN_FILE_COUNT == 38
    assert len(names) == module.PLUGIN_FILE_COUNT
    assert names == sorted(item.name for item in (root / "occ3_sglang").glob("*.py"))
    assert not set(names).intersection(module.GENERATED_OUTPUTS)


def test_label_anchor_candidate_inputs_attest_live_policy():
    import json
    root = Path(__file__).resolve().parents[1]
    packaged = json.loads((root / "deploy/native-build-inputs/native-profile-input.json.v3").read_text())
    template = json.loads((root / "deploy/v3/plan-templates/occ3-icalib-persistent-candidate.json").read_text())
    admission = template["native_validator"]["admission_template"]["source_profile"]
    for profile in (packaged, admission):
        assert "label_anchor_alpha" not in profile
        assert profile["label_anchor_alpha_allowed"] == [0.0, 0.3, 0.5, 0.7, 1.0]
        # The profile default matches the post-#265 client default (0.3), so a
        # client that omits the field is anchored at the byte-exact hybrid
        # coefficient instead of silently dropping to 0.0 - the exact garbling
        # class PR #265 fixed client-side. An explicit 0.0 request still wins.
        assert profile["label_anchor_alpha_default"] == 0.3
        assert profile["anchor_alpha"] == 0.0


def test_v3_profile_default_is_the_post_265_client_default():
    """An omitting client must be anchored at the attested hybrid 0.3.

    The wire cannot distinguish "old client" from "intends 0.0" (omission is
    accepted, explicit null is refused), so the profile default itself has to
    be the safe value. Resolving a capture intent with no label_anchor_alpha
    against the packaged v3 profile must select 0.3, never 0.0.
    """
    import json
    from dataclasses import replace
    from occ3_sglang.latent_types import RuntimeProfile
    root = Path(__file__).resolve().parents[1]
    packaged = json.loads((root / "deploy/native-build-inputs/native-profile-input.json.v3").read_text())
    static = {name: value for name, value in packaged.items()
              if name not in ("declared_identity",)}
    profile = RuntimeProfile(runtime_epoch="e" * 32, runtime_generation=1, **static)
    assert profile.resolve_label_anchor(None) == 0.3
    assert profile.resolve_label_anchor(0.0) == 0.0
    # The legacy per-message packaging default carries no policy at all; its
    # fixed zero stays a historical pinned artifact, not the live default.
    per_message = json.loads((root / "deploy/native-build-inputs/native-profile-input.json").read_text())
    assert "label_anchor_alpha_default" not in per_message
    assert "label_anchor_alpha_allowed" not in per_message
