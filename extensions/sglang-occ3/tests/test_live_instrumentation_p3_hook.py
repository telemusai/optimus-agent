"""Offline tests for the P3 arm instrumentation (unattested site-packages hook).

The hook is an operational instrument, not plugin source: it lives outside the
digest-attested `occ3_sglang` package and must stay testable without sglang, a
GPU or a DGX. These tests exercise exactly the properties the P3 brief requires
before a live window: the wrap reaches both the resolver and the name
`latent_worker` bound at import; an unwrappable resolver or an unrebindable
worker binding is refused with the shipped attribute restored; the arm comes
from a control file re-read per call; and the four documented contents (one of
three arms, or nothing) each return the declared rows.

The torch-dependent checks skip when torch is absent; everything else runs
without it.
"""
from __future__ import annotations

import importlib
import importlib.util
import itertools
import pathlib
import sys
import types

import pytest

EXT_ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOK_PATH = EXT_ROOT / "deploy" / "v3" / "live-instrumentation" / "occ3_p3_arms_hook.py"
PTH_PATH = EXT_ROOT / "deploy" / "v3" / "live-instrumentation" / "zz_occ3_p3_arms.pth"
FAKE_PACKAGE = "occ3_sglang"

_counter = itertools.count()


def load_hook():
    """A fresh hook module per test (no shared install state, no autostart)."""
    name = "occ3_p3_arms_hook_under_test_%d" % next(_counter)
    spec = importlib.util.spec_from_file_location(name, HOOK_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _purge_modules():
    for name in [n for n in sys.modules if n == FAKE_PACKAGE or n.startswith(FAKE_PACKAGE + ".")]:
        del sys.modules[name]


def _purge_finders():
    sys.meta_path[:] = [f for f in sys.meta_path if type(f).__name__ != "_PostImportFinder"]


@pytest.fixture(autouse=True)
def _isolate_real_package():
    """This file purges and re-imports the real occ3_sglang package repeatedly.

    Guarantee the exact pre-test modules and finder list after every test, so
    no wrapped module object or leftover finder leaks into later suites.
    """
    saved_modules = {n: m for n, m in sys.modules.items()
                     if n == FAKE_PACKAGE or n.startswith(FAKE_PACKAGE + ".")}
    saved_finders = list(sys.meta_path)
    yield
    _purge_modules()
    _purge_finders()
    importlib.invalidate_caches()
    sys.modules.update(saved_modules)
    sys.meta_path[:] = saved_finders


@pytest.fixture
def fake_site(tmp_path, monkeypatch):
    """A stand-in site-packages entry with the two modules the hook must reach."""
    root = tmp_path / "site"
    package = root / FAKE_PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "alignment_dispatch.py").write_text(
        "class Info:\n"
        "    def __init__(self, name):\n"
        "        self.name = name\n"
        "\n"
        "\n"
        "def aligner_for(profile, artifact=None):\n"
        "    def align(hidden, embeddings, target_norm):\n"
        "        return embeddings * 2, Info('shipped')\n"
        "    align.serving = {'kind': 'n1'}\n"
        "    align.artifact_digest = ''\n"
        "    align.rows = 7\n"
        "    return align\n",
        encoding="utf-8")
    (package / "latent_worker.py").write_text(
        "from .alignment_dispatch import aligner_for\n", encoding="utf-8")
    saved = {n: m for n, m in sys.modules.items()
             if n == FAKE_PACKAGE or n.startswith(FAKE_PACKAGE + ".")}
    _purge_modules()
    _purge_finders()
    monkeypatch.syspath_prepend(str(root))
    yield root
    _purge_modules()
    _purge_finders()
    sys.modules.update(saved)


def _write_control(path, data):
    path.write_bytes(data)
    return path


# -- install: wrap reach ------------------------------------------------------

def test_install_wraps_the_resolver_and_the_latent_worker_binding(tmp_path, fake_site):
    hook = load_hook()
    control = tmp_path / "arm"
    state = hook.install(control=str(control))
    assert state["state"] == "installed", state
    assert state["armed"] is True
    assert state["control_path"] == str(control)

    dispatch = importlib.import_module(FAKE_PACKAGE + ".alignment_dispatch")
    worker = importlib.import_module(FAKE_PACKAGE + ".latent_worker")
    assert dispatch.aligner_for.__occ3_p3_wrapped__ is True
    # The name latent_worker bound at import is the wrapped one.
    assert worker.aligner_for is dispatch.aligner_for
    assert sorted(hook.status()["patched"]) == [
        FAKE_PACKAGE + ".alignment_dispatch", FAKE_PACKAGE + ".latent_worker"]


def test_install_rebinds_a_worker_that_was_imported_first(tmp_path, fake_site):
    dispatch = importlib.import_module(FAKE_PACKAGE + ".alignment_dispatch")
    worker = importlib.import_module(FAKE_PACKAGE + ".latent_worker")
    shipped = worker.aligner_for
    hook = load_hook()
    hook.install(control=str(tmp_path / "arm"))
    assert worker.aligner_for is not shipped
    assert worker.aligner_for is dispatch.aligner_for
    assert worker.aligner_for.__occ3_p3_original__ is shipped


def test_install_is_idempotent(tmp_path, fake_site):
    hook = load_hook()
    first = hook.install(control=str(tmp_path / "arm"))
    second = hook.install(control=str(tmp_path / "other"))
    assert first["state"] == second["state"] == "installed"
    assert second["control_path"] == first["control_path"]
    assert len([f for f in sys.meta_path if type(f).__name__ == "_PostImportFinder"]) == 1


# -- install: refusal path ----------------------------------------------------

def test_install_refuses_a_resolver_from_another_module(tmp_path, monkeypatch):
    root = tmp_path / "site"
    package = root / FAKE_PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "elsewhere.py").write_text(
        "def aligner_for(profile, artifact=None):\n"
        "    return lambda hidden, embeddings, target_norm: (embeddings, None)\n",
        encoding="utf-8")
    (package / "alignment_dispatch.py").write_text(
        "from .elsewhere import aligner_for\n", encoding="utf-8")
    (package / "latent_worker.py").write_text(
        "from .alignment_dispatch import aligner_for\n", encoding="utf-8")
    _purge_modules()
    _purge_finders()
    monkeypatch.syspath_prepend(str(root))
    try:
        dispatch = importlib.import_module(FAKE_PACKAGE + ".alignment_dispatch")
        worker = importlib.import_module(FAKE_PACKAGE + ".latent_worker")
        shipped_dispatch, shipped_worker = dispatch.aligner_for, worker.aligner_for
        hook = load_hook()
        state = hook.install(control=str(tmp_path / "arm"))
        assert state["state"] == "refused", state
        assert state["code"] == "resolver_not_the_real_aligner"
        # Nothing is wrapped and no finder stays armed.
        assert dispatch.aligner_for is shipped_dispatch
        assert worker.aligner_for is shipped_worker
        assert not [f for f in sys.meta_path if type(f).__name__ == "_PostImportFinder"]
        # The refusal is sticky: a second install cannot quietly proceed.
        assert hook.install(control=str(tmp_path / "arm2"))["state"] == "refused"
    finally:
        _purge_modules()
        _purge_finders()


@pytest.mark.parametrize("body,code,shipped", [
    ("other = 1\n", "resolver_missing", None),
    ("aligner_for = None\n", "resolver_missing", None),
    ("aligner_for = 7\n", "resolver_not_callable", 7),
])
def test_install_refuses_a_missing_or_non_callable_resolver(tmp_path, monkeypatch, body, code,
                                                             shipped):
    root = tmp_path / "site"
    package = root / FAKE_PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "alignment_dispatch.py").write_text(body, encoding="utf-8")
    (package / "latent_worker.py").write_text(
        "from .alignment_dispatch import aligner_for\n", encoding="utf-8")
    _purge_modules()
    _purge_finders()
    monkeypatch.syspath_prepend(str(root))
    try:
        hook = load_hook()
        assert hook.install(control=str(tmp_path / "arm"))["state"] == "installed"
        # The refusal fires when the module the hook must wrap is imported.
        dispatch = importlib.import_module(FAKE_PACKAGE + ".alignment_dispatch")
        state = hook.status()
        assert state["state"] == "refused", state
        assert state["code"] == code
        assert getattr(dispatch, "aligner_for", None) == shipped
        assert not [f for f in sys.meta_path if type(f).__name__ == "_PostImportFinder"]
    finally:
        _purge_modules()
        _purge_finders()


def test_install_refuses_when_the_worker_binding_cannot_be_rebound(tmp_path, fake_site):
    dispatch = importlib.import_module(FAKE_PACKAGE + ".alignment_dispatch")
    importlib.import_module(FAKE_PACKAGE + ".latent_worker")
    shipped = dispatch.aligner_for

    class ReadOnlyWorker:
        @property
        def aligner_for(self):
            return shipped

    sys.modules[FAKE_PACKAGE + ".latent_worker"] = ReadOnlyWorker()
    hook = load_hook()
    state = hook.install(control=str(tmp_path / "arm"))
    assert state["state"] == "refused", state
    assert state["code"] == "worker_rebind_failed"
    # Fail closed: no partial wrap survives.
    assert dispatch.aligner_for is shipped
    assert not [f for f in sys.meta_path if type(f).__name__ == "_PostImportFinder"]


def test_install_refuses_an_invalid_control_path(monkeypatch, fake_site):
    monkeypatch.setenv("OCC3_P3_ARM_FILE", "relative/arm")
    hook = load_hook()
    state = hook.install()
    assert state["state"] == "refused"
    assert state["code"] == "arm_path_invalid"
    assert not [f for f in sys.meta_path if type(f).__name__ == "_PostImportFinder"]


def test_default_control_path_is_the_declared_one(monkeypatch, fake_site):
    monkeypatch.delenv("OCC3_P3_ARM_FILE", raising=False)
    hook = load_hook()
    assert hook.install()["control_path"] == "/tmp/occ3_p3_arm"


# -- arm selection ------------------------------------------------------------

@pytest.fixture(scope="module")
def torch():
    return pytest.importorskip("torch")


def build_proxy(hook, control, torch, rows=6, dim=6144, dtype=None):
    dtype = dtype or torch.float32
    def original(hidden, embeddings, target_norm):
        return torch.full_like(embeddings, 3.0), "info-shipped"
    original.serving = {"kind": "n1"}
    original.artifact_digest = "digest-shipped"
    original.rows = 11
    return hook._ArmAligner(original, str(control), "statebridge.test.v1")


def test_texact_returns_the_embedding_rows_unchanged(tmp_path, torch):
    hook = load_hook()
    control = _write_control(tmp_path / "arm", b"texact")
    proxy = build_proxy(hook, control, torch)
    embeddings = torch.arange(6 * 4, dtype=torch.float32).reshape(6, 4)
    rows, info = proxy(torch.zeros_like(embeddings), embeddings, torch.tensor(0.7))
    assert info is None
    assert rows is embeddings
    assert torch.equal(rows, embeddings)


def test_tnorm_renormalises_the_same_rows_to_nu(tmp_path, torch, caplog):
    hook = load_hook()
    control = _write_control(tmp_path / "arm", b"tnorm\n")
    proxy = build_proxy(hook, control, torch)
    assert proxy.serving == {"kind": "n1"}
    embeddings = torch.linspace(0.1, 1.0, 6 * 4).reshape(6, 4)
    nu = 0.699
    with caplog.at_level("WARNING", logger="occ3_p3_arms"):
        # The worker hands the aligner a 0-dim tensor, not a float.
        rows, info = proxy(torch.zeros_like(embeddings), embeddings, torch.tensor(nu))
    assert info is None
    norms = rows.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, nu), atol=1e-6)
    # Same directions, different scale: the arm is a renorm of those rows.
    cos = torch.nn.functional.cosine_similarity(rows, embeddings, dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-6)
    lines = [r.getMessage() for r in caplog.records if r.name == "occ3_p3_arms"]
    assert len(lines) == 1
    assert lines[0].startswith("occ3_p3_arm=tnorm file_sha256=")
    assert " mtime_ns=" in lines[0]


def test_nshape_returns_the_mean_row_replicated(tmp_path, torch):
    hook = load_hook()
    control = _write_control(tmp_path / "arm", b"nshape")
    proxy = build_proxy(hook, control, torch)
    assert proxy.serving is None
    embeddings = torch.linspace(0.5, 2.5, 6 * 4).reshape(6, 4)
    rows, info = proxy(torch.zeros_like(embeddings), embeddings, None)
    assert info is None
    assert rows.shape == embeddings.shape
    assert rows.dtype == embeddings.dtype
    mean = embeddings.mean(dim=0)
    for index in range(rows.shape[0]):
        assert torch.allclose(rows[index], mean, atol=1e-6)
    assert float(rows.norm(dim=-1)[0]) > 0.0


def test_absent_control_file_returns_the_shipped_aligner(tmp_path, torch):
    hook = load_hook()
    proxy = build_proxy(hook, tmp_path / "missing", torch)
    assert proxy.serving == {"kind": "n1"}  # the shipped serving config
    embeddings = torch.ones(6, 4)
    rows, info = proxy(torch.zeros_like(embeddings), embeddings, torch.tensor(0.7))
    assert info == "info-shipped"
    assert torch.allclose(rows, torch.full_like(embeddings, 3.0))


@pytest.mark.parametrize("content", [
    b"", b"\n", b"TEXACT", b"texact ", b" texact", b"texact\n\n", b"texact\nextra",
    b"tnorm\nnshape", b"n-shape", b"exact", b"x" * 64, b"y" * 65,
])
def test_any_other_control_content_returns_the_shipped_aligner(tmp_path, torch, content):
    hook = load_hook()
    control = _write_control(tmp_path / "arm", content)
    proxy = build_proxy(hook, control, torch)
    embeddings = torch.ones(6, 4)
    rows, info = proxy(torch.zeros_like(embeddings), embeddings, torch.tensor(0.7))
    assert info == "info-shipped"
    assert torch.allclose(rows, torch.full_like(embeddings, 3.0))


def test_a_directory_at_the_control_path_returns_the_shipped_aligner(tmp_path, torch):
    hook = load_hook()
    control = tmp_path / "armdir"
    control.mkdir()
    proxy = build_proxy(hook, control, torch)
    embeddings = torch.ones(6, 4)
    rows, info = proxy(torch.zeros_like(embeddings), embeddings, torch.tensor(0.7))
    assert info == "info-shipped"


def test_control_file_is_reread_on_every_call(tmp_path, torch):
    hook = load_hook()
    control = tmp_path / "arm"
    proxy = build_proxy(hook, control, torch)
    embeddings = torch.linspace(0.1, 1.0, 6 * 4).reshape(6, 4)
    nu = torch.tensor(0.699)  # what the worker measures and passes

    _write_control(control, b"texact")
    rows, _ = proxy(torch.zeros_like(embeddings), embeddings, nu)
    assert rows is embeddings

    _write_control(control, b"tnorm")          # different size: re-read
    rows, _ = proxy(torch.zeros_like(embeddings), embeddings, nu)
    norms = rows.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, 0.699), atol=1e-6)

    # Same size as the previous content, so only the mtime can invalidate.
    info = control.stat()
    _write_control(control, b"nshape")
    import os
    os.utime(control, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000))
    rows, _ = proxy(torch.zeros_like(embeddings), embeddings, nu)
    assert torch.allclose(rows[0], embeddings.mean(dim=0), atol=1e-6)
    assert torch.allclose(rows[1], rows[0], atol=1e-6)

    control.unlink()                            # removed between arms
    rows, info_rows = proxy(torch.zeros_like(embeddings), embeddings, nu)
    assert info_rows == "info-shipped"


def test_serving_tracks_the_arm_without_storing_it(tmp_path, torch):
    hook = load_hook()
    control = _write_control(tmp_path / "arm", b"texact")
    proxy = build_proxy(hook, control, torch)
    assert proxy.serving is None
    _write_control(control, b"tnorm")
    assert proxy.serving == {"kind": "n1"}
    _write_control(control, b"nshape")
    assert proxy.serving is None
    control.unlink()
    assert proxy.serving == {"kind": "n1"}


def test_proxy_forwards_the_artifact_identity(tmp_path, torch):
    hook = load_hook()
    proxy = build_proxy(hook, tmp_path / "arm", torch)
    assert proxy.artifact_digest == "digest-shipped"
    assert proxy.rows == 11


def test_one_log_line_per_aligner_call(tmp_path, torch, caplog):
    hook = load_hook()
    control = _write_control(tmp_path / "arm", b"texact")
    proxy = build_proxy(hook, control, torch)
    embeddings = torch.ones(6, 4)
    with caplog.at_level("WARNING", logger="occ3_p3_arms"):
        for _ in range(3):
            proxy(torch.zeros_like(embeddings), embeddings, None)
    lines = [r.getMessage() for r in caplog.records if r.name == "occ3_p3_arms"]
    assert len(lines) == 3
    assert all(line.startswith("occ3_p3_arm=texact file_sha256=") for line in lines)


def test_install_refuses_when_latent_worker_binds_its_own_name(tmp_path, monkeypatch):
    """A worker that keeps its own `aligner_for` cannot be instrumented."""
    root = tmp_path / "site"
    package = root / FAKE_PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "alignment_dispatch.py").write_text(
        "def aligner_for(profile, artifact=None):\n"
        "    return lambda hidden, embeddings, target_norm: (embeddings, None)\n",
        encoding="utf-8")
    (package / "latent_worker.py").write_text(
        "def aligner_for(profile, artifact=None):\n"
        "    return lambda hidden, embeddings, target_norm: (embeddings, None)\n",
        encoding="utf-8")
    _purge_modules()
    _purge_finders()
    monkeypatch.syspath_prepend(str(root))
    try:
        hook = load_hook()
        assert hook.install(control=str(tmp_path / "arm"))["state"] == "installed"
        worker = importlib.import_module(FAKE_PACKAGE + ".latent_worker")
        state = hook.status()
        assert state["state"] == "refused", state
        assert state["code"] == "worker_binding_not_wrapped"
        assert getattr(worker.aligner_for, "__occ3_p3_wrapped__", False) is False
    finally:
        _purge_modules()
        _purge_finders()


def test_tnorm_declines_without_a_usable_norm_statistic(tmp_path, torch):
    hook = load_hook()
    control = _write_control(tmp_path / "arm", b"tnorm")
    proxy = build_proxy(hook, control, torch)
    embeddings = torch.ones(6, 4)
    for bad in (None, torch.tensor(-1.0), torch.tensor(float("nan")), torch.tensor(0.0),
                "0.7", True):
        with pytest.raises(RuntimeError) as raised:
            proxy(torch.zeros_like(embeddings), embeddings, bad)
        assert str(raised.value) == "p3_arm_norm_unavailable"


def test_tnorm_accepts_the_tensor_the_worker_measures(tmp_path, torch):
    """Regression pin: mean_embedding_norm returns a 0-dim tensor, and the
    shipped N1 serving_transform refuses a tensor with `profile`. The arm
    converts it, so the T-norm turn cannot fail on the statistic's type."""
    hook = load_hook()
    control = _write_control(tmp_path / "arm", b"tnorm")
    proxy = build_proxy(hook, control, torch)
    embeddings = torch.linspace(0.2, 1.4, 6 * 4).reshape(6, 4)
    measured = (embeddings.norm(dim=-1).mean()).float()
    rows, _info = proxy(torch.zeros_like(embeddings), embeddings, measured)
    norms = rows.norm(dim=-1)
    assert torch.allclose(norms, torch.full_like(norms, float(measured)), atol=1e-5)


# -- the shipped package ------------------------------------------------------

def test_install_wraps_the_shipped_modules_and_serves_every_arm(tmp_path, torch):
    """The same wrap against the real occ3_sglang modules, offline, on CPU."""
    real_root = str(EXT_ROOT)
    saved = {n: m for n, m in sys.modules.items()
             if n == FAKE_PACKAGE or n.startswith(FAKE_PACKAGE + ".")}
    if real_root not in sys.path:
        sys.path.insert(0, real_root)
    _purge_modules()
    _purge_finders()
    try:
        hook = load_hook()
        control = tmp_path / "arm"
        state = hook.install(control=str(control))
        assert state["state"] == "installed", state
        dispatch = importlib.import_module(FAKE_PACKAGE + ".alignment_dispatch")
        worker = importlib.import_module(FAKE_PACKAGE + ".latent_worker")
        assert dispatch.aligner_for.__occ3_p3_wrapped__ is True
        assert worker.aligner_for is dispatch.aligner_for
        shipped = dispatch.aligner_for.__occ3_p3_original__
        assert shipped.__module__ == FAKE_PACKAGE + ".alignment_dispatch"
        assert shipped.__qualname__ == "aligner_for"

        from occ3_sglang.latent_types import RuntimeProfile
        profile = RuntimeProfile(runtime_epoch="epoch-offline-1", runtime_generation=1,
                                 model_fingerprint="sha256:" + "0" * 63 + "1",
                                 model_id="glm-5.3")
        # The name the worker calls, not just the module attribute.
        proxy = worker.aligner_for(profile)
        assert type(proxy).__name__ == "_ArmAligner"
        assert proxy.artifact_digest == ""

        hidden = torch.randn(4, 6144)
        embeddings = torch.randn(4, 6144)
        nu = float(embeddings.norm(dim=-1).mean())
        shipped_aligner = shipped(profile)
        reference, _info = shipped_aligner(hidden, embeddings, nu)

        rows, _ = proxy(hidden, embeddings, nu)
        assert torch.allclose(rows, reference)  # no arm file: shipped output

        control.write_bytes(b"texact")
        rows, _ = proxy(hidden, embeddings, nu)
        assert rows is embeddings

        control.write_bytes(b"tnorm")
        rows, _ = proxy(hidden, embeddings, nu)
        assert torch.allclose(rows.norm(dim=-1), torch.full((4,), nu), atol=1e-3)

        control.write_bytes(b"nshape")
        rows, _ = proxy(hidden, embeddings, nu)
        assert torch.allclose(rows[0], embeddings.mean(dim=0), atol=1e-6)
        assert torch.allclose(rows[3], rows[0], atol=1e-6)
    finally:
        _purge_modules()
        _purge_finders()
        sys.modules.update(saved)


# -- import and start-up behaviour -------------------------------------------

def test_hook_module_import_is_inert(tmp_path):
    import subprocess
    script = (
        "import importlib.util, sys\n"
        "spec = importlib.util.spec_from_file_location('occ3_p3_arms_hook_probe', %r)\n"
        "module = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(module)\n"
        "print('occ3_sglang' in sys.modules, 'torch' in sys.modules, module.status()['state'])\n"
        % str(HOOK_PATH))
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=180)
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["False", "False", "pending"], out.stdout


def test_pth_file_imports_the_hook():
    assert PTH_PATH.read_bytes() == b"import occ3_p3_arms_hook\n"


def test_pth_arms_the_hook_at_interpreter_start(tmp_path):
    """A site dir with the two extras arms the hook and wraps both names."""
    import shutil
    import subprocess
    extras = tmp_path / "extras"
    extras.mkdir()
    shutil.copy(HOOK_PATH, extras / HOOK_PATH.name)
    shutil.copy(PTH_PATH, extras / PTH_PATH.name)
    package_root = tmp_path / "site"
    package = package_root / FAKE_PACKAGE
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "alignment_dispatch.py").write_text(
        "def aligner_for(profile, artifact=None):\n"
        "    return lambda hidden, embeddings, target_norm: (embeddings, None)\n",
        encoding="utf-8")
    (package / "latent_worker.py").write_text(
        "from .alignment_dispatch import aligner_for\n", encoding="utf-8")
    script = (
        "import sys, site\n"
        "site.addsitedir(%r)\n"
        "print('imported', 'occ3_p3_arms_hook' in sys.modules)\n"
        "print('armed', sum(1 for f in sys.meta_path if type(f).__name__ == '_PostImportFinder'))\n"
        "sys.path.insert(0, %r)\n"
        "import occ3_sglang.latent_worker as worker\n"
        "import occ3_sglang.alignment_dispatch as dispatch\n"
        "print('wrapped', worker.aligner_for is dispatch.aligner_for,\n"
        "      getattr(worker.aligner_for, '__occ3_p3_wrapped__', False))\n"
        % (str(extras), str(package_root)))
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                         timeout=180, env={"PATH": "", "SYSTEMROOT": "C:\\Windows"})
    assert out.returncode == 0, out.stderr
    assert out.stdout.split("\n")[0].split() == ["imported", "True"], out.stdout
    assert out.stdout.split("\n")[1].split() == ["armed", "1"], out.stdout
    assert out.stdout.split("\n")[2].split() == ["wrapped", "True", "True"], out.stdout
