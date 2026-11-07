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


def test_label_anchor_candidate_inputs_select_half_blend():
    import json
    root = Path(__file__).resolve().parents[1]
    packaged = json.loads((root / "deploy/native-build-inputs/native-profile-input.json.v3").read_text())
    template = json.loads((root / "deploy/v3/plan-templates/occ3-icalib-persistent-candidate.json").read_text())
    admission = template["native_validator"]["admission_template"]["source_profile"]
    for profile in (packaged, admission):
        assert profile["label_anchor_alpha"] == 0.5
        assert profile["anchor_alpha"] == 0.0
