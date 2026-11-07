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
