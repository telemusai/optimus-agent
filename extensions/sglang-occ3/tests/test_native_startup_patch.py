"""CPU-only source/startup tests. No socket, SGLang import, model or GPU use.

Real installed Uvicorn methods are inspected but never run. Loop/order tests use
explicit Uvicorn-shaped stubs; source overlay tests compile/read only, never apply
changes to the supplied checkout.
"""
from __future__ import annotations

import asyncio
import ast
import hashlib
import json
import logging
import os
import re
import stat
import types
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI

from deploy.v3 import source_patch
from occ3_sglang import native_install as install
from test_native_install import KEY, pinned_auth

BEFORE = "1fbb46671e86d0ebf23046efa09e8e03d251754e681ba03b818dbb7a75c79c55"
AFTER = "07838ebd49c9bbfa09369ffc55f001a089699769b0b6df22b96e7ba05863f57c"
GLM_BEFORE = "61a62420d07e3986ae36d0f6fa44b9232f4c331ebd8fc3b32ba0889523bdfb0e"
GLM_AFTER = "e09a129d5872c532ccdc085585cfd8fac336c73d13d09fa8c2304a79500be271"
EXT = Path(__file__).resolve().parents[1]


def source_root():
    # Raw source bytes copied by the parent from the running original. The
    # dependency checkout may have CRLF; it is not silently normalized here.
    return Path(os.environ.get("OCC3_SGLANG_ATTESTED_SOURCE", str(Path(__file__).resolve().parents[4] /
        "Optimus-Assistant/profile/workspace/occ3-campaign/v3-controlled-deploy/"
        "original-sglang-source-attestation-v2")))


def test_exact_startup_patch_prepares_without_mutating_original_source():
    manifest = source_patch.load_manifest(EXT / "patches/native-startup.v2.json")
    target = source_root() / "entrypoints/http_server.py"
    before = target.read_bytes()
    # Supplied checkout files have LF content; no normalization or fuzzy patch.
    plan = source_patch.prepare(source_root(), manifest)
    # Two-path manifest: the original http_server startup replacement stays
    # byte-identical; the parser overlay only adds the strict-stream flag.
    assert [f.relative for f in plan.files] == [
        "entrypoints/http_server.py", "function_call/glm47_moe_detector.py"]
    assert hashlib.sha256(before).hexdigest() == BEFORE
    assert hashlib.sha256(plan.files[0].patched).hexdigest() == AFTER
    assert target.read_bytes() == before
    tree = ast.parse(plan.files[0].patched)
    setup = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "_setup_and_run_http_server")
    calls = [n for n in ast.walk(setup) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "prepare_selected_native_server"]
    assert len(calls) == 1
    assert [n.id for n in calls[0].args] == ["server", "app", "tokenizer_manager"]
    # Same block, directly before native server.run(); no wrapper run or new loop.
    containing = []
    for n in ast.walk(setup):
        for _, value in ast.iter_fields(n):
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ast.Expr) and item.value is calls[0]:
                        containing.append(value[i + 1])
    assert len(containing) == 1
    next_call = containing[0].value
    assert isinstance(next_call.func, ast.Attribute) and next_call.func.attr == "run"
    assert isinstance(next_call.func.value, ast.Name) and next_call.func.value.id == "server"
    # The Granian branch carries its own additive bootstrap: exactly one call
    # inside _run_granian_server's single-tokenizer sub-branch, directly
    # before uvloop.run(serve()), with the same three in-scope arguments.
    granian = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "_run_granian_server")
    granian_calls = [n for n in ast.walk(granian) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Name)
                     and n.func.id == "prepare_selected_native_granian_server"]
    assert len(granian_calls) == 1
    assert [n.id for n in granian_calls[0].args] == ["server", "app", "tokenizer_manager"]
    granian_containing = []
    for n in ast.walk(granian):
        for _, value in ast.iter_fields(n):
            if isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, ast.Expr) and item.value is granian_calls[0]:
                        granian_containing.append(value[i + 1])
    assert len(granian_containing) == 1
    run_call = granian_containing[0].value
    assert isinstance(run_call.func, ast.Attribute) and run_call.func.attr == "run"
    assert isinstance(run_call.func.value, ast.Name) and run_call.func.value.id == "uvloop"
    assert (isinstance(run_call.args[0], ast.Call)
            and isinstance(run_call.args[0].func, ast.Name)
            and run_call.args[0].func.id == "serve")
    # The two flavors never cross: no uvicorn-flavor call inside the Granian
    # helper and no Granian-flavor call inside the uvicorn setup helper.
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "prepare_selected_native_server"
                   for n in ast.walk(granian))
    assert not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == "prepare_selected_native_granian_server"
                   for n in ast.walk(setup))
    # Parser overlay: strict flag default False, one except-clause guard, no other drift.
    glm = next(f for f in plan.files if f.relative == "function_call/glm47_moe_detector.py")
    assert hashlib.sha256(glm.original).hexdigest() == GLM_BEFORE
    assert hashlib.sha256(glm.patched).hexdigest() == GLM_AFTER
    patched_text = glm.patched.decode("utf-8")
    assert "contextvars" in patched_text and "_OCC3_STRICT_STREAM" in patched_text
    assert patched_text.count("if _OCC3_STRICT_STREAM.get():\n                raise") == 1
    # The legacy fallback line is preserved for ordinary serving.
    assert "return StreamingParseResult(normal_text=current_text)" in patched_text
    base_text = glm.original.decode("utf-8")
    assert base_text.count("return StreamingParseResult(normal_text=current_text)") == 1


def test_granian_http2_branch_bootstraps_through_granian_entry_point_only():
    # The 2026-11-03 live incident proved an --enable-http2 boot takes
    # _run_granian_server and never reaches the uvicorn default branch, so the
    # v1 bootstrap call never ran. The v2 manifest adds an additive call inside
    # _run_granian_server. Contract: the http2 branch references exactly the
    # granian entry point -- never the uvicorn-flavor call, never a direct
    # install_native/occ3 import beyond that one reviewed call.
    manifest = source_patch.load_manifest(EXT / "patches/native-startup.v2.json")
    plan = source_patch.prepare(source_root(), manifest)
    patched = plan.files[0].patched.decode("utf-8")
    tree = ast.parse(patched)
    setup = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name == "_setup_and_run_http_server")

    def branch_of(node):
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.If):
                yield child
            yield from branch_of(child)

    granian_branch = None
    for check in branch_of(setup):
        test = check.test
        if (isinstance(test, ast.Attribute) and test.attr == "enable_http2"):
            granian_branch = check
            break
    assert granian_branch is not None, "enable_http2 branch not found in patched source"
    # .body only: the elif/else chain (orelse) legitimately contains the
    # default uvicorn branch with its own bootstrap call.
    body_names = set()
    for node in granian_branch.body:
        body_names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
        body_names |= {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    assert "_run_granian_server" in body_names
    assert "prepare_selected_native_server" not in body_names
    assert not any(name.startswith("occ3") for name in body_names)
    # The uvicorn bootstrap call stays exactly once, in the default branch.
    calls = [n for n in ast.walk(setup) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)
             and n.func.id == "prepare_selected_native_server"]
    assert len(calls) == 1
    # The granian entry point exists module-wide exactly once, inside
    # _run_granian_server, and the branch never bypasses it.
    granian = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                   and n.name == "_run_granian_server")
    module_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                    and isinstance(n.func, ast.Name)
                    and n.func.id == "prepare_selected_native_granian_server"]
    assert len(module_calls) == 1
    assert module_calls[0] in [c for c in ast.walk(granian) if c is module_calls[0]]
    granian_names = {n.id for n in ast.walk(granian) if isinstance(n, ast.Name)}
    granian_names |= {n.attr for n in ast.walk(granian) if isinstance(n, ast.Attribute)}
    assert "install_native_runtime" not in granian_names
    # No other occ3 reference exists in the whole patched file.
    assert patched.count("occ3_sglang") == 2  # one import per entry point
    imports = [n for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)
               and n.module == "occ3_sglang.native_install"]
    assert len(imports) == 2


def _install_glm47_chain():
    """Install the prepared patched detector + real snapshot deps into sys.modules.

    Returns the patched module. Caller owns module-state cleanup; nothing here
    imports a real SGLang runtime or touches a real checkout.
    """
    import sys, types as t
    manifest = source_patch.load_manifest(EXT / "patches/native-startup.v2.json")
    root = source_root()
    target = root / "function_call" / "glm47_moe_detector.py"
    assert hashlib.sha256(target.read_bytes()).hexdigest() == GLM_BEFORE
    plan = source_patch.prepare(root, manifest)
    plan_glm = next(f for f in plan.files if f.relative == "function_call/glm47_moe_detector.py")
    import json as _json
    orjson = t.ModuleType("orjson"); orjson.loads = _json.loads; orjson.dumps = lambda o, **k: _json.dumps(o).encode()
    sys.modules["orjson"] = orjson
    pjp = t.ModuleType("partial_json_parser"); pjp.loads = lambda s, flags=None: _json.loads(s)
    class _PE(Exception): pass
    pjp.ParseError = _PE; sys.modules["partial_json_parser"] = pjp
    pjp.__path__ = []  # namespace package so submodule imports resolve
    core = t.ModuleType("partial_json_parser.core"); core.__path__ = []
    sys.modules["partial_json_parser.core"] = core
    exceptions = t.ModuleType("partial_json_parser.core.exceptions")
    class MalformedJSON(ValueError): pass
    exceptions.MalformedJSON = MalformedJSON; core.exceptions = exceptions
    sys.modules["partial_json_parser.core.exceptions"] = exceptions
    options = t.ModuleType("partial_json_parser.core.options")
    class Allow: ALL = 0; STR = 1
    options.Allow = Allow; core.options = options
    sys.modules["partial_json_parser.core.options"] = options

    sglang = t.ModuleType("sglang"); sglang.__path__ = []; sys.modules["sglang"] = sglang
    srt = t.ModuleType("sglang.srt"); srt.__path__ = []; sys.modules["sglang.srt"] = srt
    entry = t.ModuleType("sglang.srt.entrypoints"); entry.__path__ = []; sys.modules["sglang.srt.entrypoints"] = entry
    openai_m = t.ModuleType("sglang.srt.entrypoints.openai"); openai_m.__path__ = []; sys.modules["sglang.srt.entrypoints.openai"] = openai_m
    protocol = t.ModuleType("sglang.srt.entrypoints.openai.protocol")
    class Tool:
        def __init__(self, function=None, **kw): self.function = function
    class ToolChoice:
        def __init__(self, function=None, **kw): self.function = function
    protocol.Tool = Tool; protocol.ToolChoice = ToolChoice
    openai_m.protocol = protocol
    sys.modules["sglang.srt.entrypoints.openai.protocol"] = protocol
    environ = t.ModuleType("sglang.srt.environ")
    class _Envs:
        def get(self, default=None): return default
    environ.envs = _Envs(); sys.modules["sglang.srt.environ"] = environ
    fc = t.ModuleType("sglang.srt.function_call"); fc.__path__ = []; sys.modules["sglang.srt.function_call"] = fc

    def load_real(rel, name):
        src = (root / rel).read_text(encoding="utf-8")
        m = t.ModuleType(name); m.__file__ = str(root / rel)
        sys.modules[name] = m
        exec(compile(src, m.__file__, "exec"), m.__dict__)
        return m
    load_real("function_call/core_types.py", "sglang.srt.function_call.core_types")
    load_real("function_call/utils.py", "sglang.srt.function_call.utils")
    load_real("function_call/base_format_detector.py", "sglang.srt.function_call.base_format_detector")
    glm = t.ModuleType("sglang.srt.function_call.glm47_moe_detector")
    glm.__file__ = "glm47_moe_detector.py"
    sys.modules["sglang.srt.function_call.glm47_moe_detector"] = glm
    exec(compile(plan_glm.patched.decode("utf-8"), glm.__file__, "exec"), glm.__dict__)
    return glm


def _load_patched_glm47():
    """Install, yield-then-restore wrapper used by the overlay unit tests."""
    import sys
    saved = {k: v for k, v in sys.modules.items()
             if k.startswith(("sglang", "orjson", "partial_json_parser", "xgrammar"))}
    glm = _install_glm47_chain()
    return glm, sys.modules["sglang.srt.entrypoints.openai.protocol"].Tool, None


def _restore_glm47_chain():
    import sys
    for key in [k for k in sys.modules if k.startswith(("sglang", "orjson", "partial_json_parser", "xgrammar"))]:
        del sys.modules[key]


BOT = "<tool_call" + ">"
EOT = "</tool_call" + ">"
AKO = "<arg_key" + ">"
AKC = "</arg_key" + ">"
AVO = "<arg_value" + ">"
AVC = "</arg_value" + ">"


def _strict_probe_tool(Tool):
    schema = {"type": "object", "properties": {"x": {"type": "number"}},
              "oneOf": [{"properties": {"kind": {"const": "a"}, "y": {"type": "string"}}},
                        {"properties": {"kind": {"const": "b"}, "y": {"type": "number"}}}]}
    return [Tool(function=type("F", (), {"name": "probe", "parameters": schema, "strict": False})())]


def _strict_probe_text():
    return BOT + "probe" + AKO + "x" + AKC + AVO + "b'hi'" + AVC + EOT


def test_overlay_legacy_bytes_unchanged_and_strict_raises():
    """The frozen 76-byte counterexample: legacy recurrence, strict fail-once."""
    glm, Tool, _ = _load_patched_glm47()
    tools = _strict_probe_tool(Tool)
    text = _strict_probe_text()
    assert len(text) == 76
    det = glm.Glm47MoeDetector()
    emitted = []
    for chunk in [text, "a", "b", "c", "d"]:
        result = det.parse_streaming_increment(chunk, tools)
        emitted.append(len(result.normal_text) if result.normal_text else 0)
    # Exact legacy recurrence from the frozen report: every byte re-emitted per update.
    assert emitted == [76, 77, 78, 79, 80]
    assert len(det._buffer) == 80

    det = glm.Glm47MoeDetector()
    token = glm._OCC3_STRICT_STREAM.set(True)
    try:
        with pytest.raises(TypeError, match="not JSON serializable"):
            det.parse_streaming_increment(text, tools)
    finally:
        glm._OCC3_STRICT_STREAM.reset(token)
    assert glm._OCC3_STRICT_STREAM.get() is False
    _restore_glm47_chain()


def test_overlay_strict_raises_value_error_variant_and_resets():
    """The original exception may itself be ValueError; strict must re-raise it."""
    glm, Tool, _ = _load_patched_glm47()
    tools = _strict_probe_tool(Tool)
    text = _strict_probe_text()
    original = glm.Glm47MoeDetector._send_tool_name_if_needed
    def boom(self, *args, **kwargs): raise ValueError("synthetic value error")
    glm.Glm47MoeDetector._send_tool_name_if_needed = boom
    try:
        det = glm.Glm47MoeDetector()
        result = det.parse_streaming_increment(text, tools)
        # Legacy fallback: swallowed, whole current_text returned as normal text.
        assert result.normal_text == text
        token = glm._OCC3_STRICT_STREAM.set(True)
        try:
            det = glm.Glm47MoeDetector()
            with pytest.raises(ValueError, match="synthetic value error"):
                det.parse_streaming_increment(text, tools)
        finally:
            glm._OCC3_STRICT_STREAM.reset(token)
        # After reset, an ordinary call takes the legacy path again.
        det = glm.Glm47MoeDetector()
        result = det.parse_streaming_increment("plain text", tools)
        assert result.normal_text == "plain text"
    finally:
        glm.Glm47MoeDetector._send_tool_name_if_needed = original
    _restore_glm47_chain()


def test_overlay_concurrent_ordinary_request_unaffected():
    """Ordinary requests never set the flag; a concurrent opted step cannot leak it."""
    glm, Tool, _ = _load_patched_glm47()
    tools = _strict_probe_tool(Tool)
    text = _strict_probe_text()
    # Ordinary detector in default context: legacy fallback, no raise.
    det = glm.Glm47MoeDetector()
    result = det.parse_streaming_increment(text, tools)
    assert result.normal_text == text
    # Set in one task must not be visible in another task's context copy.
    async def opted():
        token = glm._OCC3_STRICT_STREAM.set(True)
        try:
            await asyncio.sleep(0)
            glm.Glm47MoeDetector().parse_streaming_increment(text, tools)
            return "no-raise"
        except TypeError:
            return "raised"
        finally:
            glm._OCC3_STRICT_STREAM.reset(token)
    async def ordinary():
        await asyncio.sleep(0)
        result = glm.Glm47MoeDetector().parse_streaming_increment(text, tools)
        return len(result.normal_text or "")
    async def main():
        return await asyncio.gather(opted(), ordinary())
    raised, ordinary_len = asyncio.run(main())
    assert raised == "raised" and ordinary_len == 76
    assert glm._OCC3_STRICT_STREAM.get() is False
    _restore_glm47_chain()



@pytest.mark.parametrize("fault", ["hash", "anchor", "afterhash"])
def test_patch_rejects_wrong_original_or_fuzzy_or_unpinned_result(tmp_path, fault):
    manifest = source_patch.load_manifest(EXT / "patches/native-startup.v2.json")
    data = (source_root() / "entrypoints/http_server.py").read_bytes()
    dest = tmp_path / "entrypoints/http_server.py"
    dest.parent.mkdir()
    if fault == "hash": data += b"# unrelated drift\n"
    elif fault == "anchor": manifest["files"][0]["replacements"][0]["before"] += "absent"
    else: manifest["files"][0]["after_sha256"] = "0" * 64
    dest.write_bytes(data)
    with pytest.raises(source_patch.PatchError):
        source_patch.prepare(tmp_path, manifest)
    assert dest.read_bytes() == data


@pytest.fixture
def fixed_selection_file(tmp_path, monkeypatch):
    # Explicit filesystem metadata model for Windows; real POSIX behavior remains
    # covered by the profile owner, not asserted by these tests.
    monkeypatch.setattr(install, "__file__", str(tmp_path / "native_install.py"))
    original = os.fstat
    def private_stat(fd):
        result = original(fd)
        fields = {name: getattr(result, name) for name in
            ("st_mode", "st_nlink", "st_size", "st_dev", "st_ino", "st_mtime_ns")}
        fields["st_mode"] = (fields["st_mode"] & ~0o777) | 0o600
        return NS(**fields)
    monkeypatch.setattr(os, "fstat", private_stat)
    return tmp_path / "_native_profile_digest.py"


def selected_text(selected="True"):
    return ("# generated literals only\nPROFILE_SHA256 = '" + "a" * 64
            + "'\nSOURCE_SHA256 = '" + "b" * 64
            + "'\nEXPERIMENTAL_STARTUP = " + selected + "\n")


def test_fixed_build_selection_absent_false_and_true(fixed_selection_file):
    path = fixed_selection_file
    assert install._read_startup_literal() is None
    path.write_text(selected_text("False"))
    assert install._read_startup_literal() is None
    path.write_text(selected_text())
    assert install._read_startup_literal() == {"PROFILE_SHA256": "a" * 64,
        "SOURCE_SHA256": "b" * 64, "EXPERIMENTAL_STARTUP": True}


@pytest.mark.parametrize("content", ["import os\n", "raise Exception('PRIVATE')\n",
    "EXPERIMENTAL_STARTUP = 1\n", "EXPERIMENTAL_STARTUP = True\n", selected_text() + "OTHER = 1\n",
    selected_text() + "SOURCE_SHA256 = 'changed'\n", selected_text().replace("a" * 64, "wrong"),
    selected_text("bool(1)"), selected_text() + "#" * 5000])
def test_selection_only_accepts_bounded_exact_literals_without_execution(fixed_selection_file, content):
    fixed_selection_file.write_text(content)
    with pytest.raises(install.InstallationError) as caught:
        install._read_startup_literal()
    assert "PRIVATE" not in str(caught.value)


def test_selection_rejects_symlink_boundary(fixed_selection_file, monkeypatch):
    fixed_selection_file.write_text(selected_text())
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda p: p == fixed_selection_file or original(p))
    with pytest.raises(install.InstallationError, match="startup_selection_unsafe"):
        install._read_startup_literal()


def native_server_fixture():
    import uvicorn
    app = FastAPI()
    args = NS(tokenizer_worker_num=1, enable_http2=False, enable_ssl_refresh=False)
    app.server_args = args
    app.is_single_tokenizer_mode = True
    tm = NS(server_args=args)
    server = uvicorn.Server(uvicorn.Config(app, loop="uvloop", log_config=None))
    return server, app, tm


def test_real_installed_uvicorn_provenance_no_run_or_sockets(monkeypatch):
    server, app, tm = native_server_fixture()
    import uvicorn.server
    if Path(uvicorn.server.__file__).stat().st_nlink != 1:
        # This Windows uv-managed environment hardlinks package files to cache.
        # Verify real refusal, then explicitly model an image's single-link file
        # to check real loaded function bytecode. This is not Linux layout proof.
        with pytest.raises(install.InstallationError, match="startup_server_provenance"):
            install._verify_native_server(server, app, tm)
        original_fstat = os.fstat
        def single_link(fd):
            result = original_fstat(fd)
            fields = {name: getattr(result, name) for name in
                ("st_mode", "st_size", "st_dev", "st_ino", "st_mtime_ns")}
            return NS(**fields, st_nlink=1)
        monkeypatch.setattr(os, "fstat", single_link)
    original = install._verify_native_server(server, app, tm)
    assert original.__self__ is server and original.__func__ is type(server).serve
    assert "serve" not in vars(server) and not server.started


@pytest.mark.parametrize("fault", ["loaded", "worker", "reload", "loop", "root", "http2", "ssl-refresh",
                                  "multi-tokenizer", "wrong-app", "stack", "instance-method"])
def test_real_server_refuses_unsupported_branch_or_existing_patch(fault):
    server, app, tm = native_server_fixture()
    if fault == "loaded": server.config.loaded = True
    elif fault == "worker": server.config.workers = 2
    elif fault == "reload": server.config.reload = True
    elif fault == "loop": server.config.loop = "asyncio"
    elif fault == "root": server.config.root_path = "/other"
    elif fault == "http2": tm.server_args.enable_http2 = True
    elif fault == "ssl-refresh": tm.server_args.enable_ssl_refresh = True
    elif fault == "multi-tokenizer": tm.server_args.tokenizer_worker_num = 2
    elif fault == "wrong-app": server.config.app = FastAPI()
    elif fault == "stack": app.middleware_stack = object()
    else: server.serve = lambda: None
    with pytest.raises(install.InstallationError, match="unsupported_native_startup"):
        install._verify_native_server(server, app, tm)


@pytest.fixture
def loop_server(monkeypatch, pinned_auth):
    from occ3_sglang import startup_state

    app = FastAPI()
    tm = NS(server_args=NS(api_key=KEY))
    pinned_auth.add_api_key_middleware(app, api_key=KEY, admin_api_key=None)
    events = []
    class Server:
        should_exit = False
        async def serve(self, *args, **kwargs):
            events.append(("serve", asyncio.get_running_loop(), args, kwargs))
            assert app.middleware_stack is None
            app.middleware_stack = "native-serve-built-stack"
            return "native-result"
    server = Server()
    monkeypatch.setattr(install, "_read_startup_literal", lambda: {"EXPERIMENTAL_STARTUP": True})
    monkeypatch.setattr(install, "_verify_startup_artifacts", lambda selection: events.append(("artifacts",)))
    monkeypatch.setattr(install, "_verify_native_server", lambda candidate, a, t: candidate.serve)
    # The attested control-plane module is the real checked-out source here.
    module_file = Path(startup_state.__file__).resolve()
    monkeypatch.setattr(install, "_verify_control_plane_sources",
                        lambda: {str(module_file): module_file.read_text(encoding="utf-8")})
    async def factory(a, t, **kwargs):
        assert a is app and t is tm and kwargs == {"experimental_opt_in": True}
        assert app.middleware_stack is None
        events.append(("install", asyncio.get_running_loop()))
        return install.InstallationReceipt(True, "installed_experimental_not_live_validated")
    monkeypatch.setattr(install, "install_native_runtime", factory)
    return NS(app=app, tm=tm, server=server, events=events, factory=factory,
              startup_state=startup_state)


@pytest.fixture(autouse=True)
def clean_startup_state():
    from occ3_sglang import startup_state

    startup_state.reset()
    yield
    startup_state.reset()


async def test_instance_wrapper_awaits_install_on_native_loop_before_serve(loop_server):
    f = loop_server
    original_type_serve = type(f.server).serve
    receipt = install.prepare_selected_native_server(f.server, f.app, f.tm)
    assert receipt.prepared and not receipt.evidence()["live_proof"]
    assert [event[0] for event in f.events] == ["artifacts"]
    assert type(f.server).serve is original_type_serve
    f.server.should_exit = True  # Original stop state is not reset.
    result = await f.server.serve("positional", sockets=["unchanged"])
    assert result == "native-result" and f.server.should_exit is True
    assert [event[0] for event in f.events] == ["artifacts", "install", "serve"]
    assert f.events[1][1] is f.events[2][1] is asyncio.get_running_loop()
    assert f.events[2][2:] == (("positional",), {"sockets": ["unchanged"]})
    assert not getattr(f.app, install._STARTUP_ATTR)["gate8_pass"]


def test_absent_selection_preserves_original_method_and_does_no_profile_work(loop_server, monkeypatch):
    f = loop_server
    monkeypatch.setattr(install, "_read_startup_literal", lambda: None)
    original = f.server.serve
    receipt = install.prepare_selected_native_server(f.server, f.app, f.tm)
    assert not receipt.selected and f.events == [] and f.server.serve == original
    assert "serve" not in vars(f.server)


def test_selected_bad_artifact_refuses_before_mutation(loop_server, monkeypatch):
    f = loop_server
    def fail(selection): raise RuntimeError("PRIVATE-PROFILE-PATH")
    monkeypatch.setattr(install, "_verify_startup_artifacts", fail)
    result = install.prepare_selected_native_server(f.server, f.app, f.tm)
    assert not result.prepared and "PRIVATE" not in str(result.evidence())
    assert "serve" not in vars(f.server) and not hasattr(f.server, install._SERVER_ATTR)


async def test_duplicate_prepare_idempotent_but_run_is_one_shot(loop_server):
    f = loop_server
    install.prepare_selected_native_server(f.server, f.app, f.tm)
    wrapper = f.server.serve
    assert install.prepare_selected_native_server(f.server, f.app, f.tm).prepared
    assert f.server.serve is wrapper and len(f.events) == 1
    await wrapper()
    with pytest.raises(install.InstallationError, match="startup_owner_mismatch"):
        await wrapper()


@pytest.mark.parametrize("code", ["backend_not_ready", "source_digest_mismatch", "installation_unavailable"])
async def test_verified_safe_refusal_can_keep_original_ordinary_startup(loop_server, monkeypatch, code):
    f = loop_server
    async def refuse(*args, **kwargs): return install.InstallationReceipt(False, code)
    monkeypatch.setattr(install, "install_native_runtime", refuse)
    install.prepare_selected_native_server(f.server, f.app, f.tm)
    assert await f.server.serve() == "native-result"
    assert getattr(f.app, install._STARTUP_ATTR)["code"] == code


@pytest.mark.parametrize("code", ["installation_cleanup_unconfirmed", "factory_cleanup_unconfirmed",
    "installation_busy", "installation_owner_mismatch", "control_client_occupied", "hook_target_occupied"])
async def test_uncertain_refusal_never_starts_serving(loop_server, monkeypatch, code):
    f = loop_server
    async def refuse(*args, **kwargs): return install.InstallationReceipt(False, code)
    monkeypatch.setattr(install, "install_native_runtime", refuse)
    install.prepare_selected_native_server(f.server, f.app, f.tm)
    with pytest.raises(install.InstallationError, match="native_startup_cleanup_unconfirmed"):
        await f.server.serve()
    assert not any(event[0] == "serve" for event in f.events)


async def test_residual_control_listener_blocks_even_otherwise_safe_refusal(loop_server, monkeypatch):
    f = loop_server
    f.tm._occ3_control_client = object()
    async def refuse(*args, **kwargs): return install.InstallationReceipt(False, "backend_not_ready")
    monkeypatch.setattr(install, "install_native_runtime", refuse)
    install.prepare_selected_native_server(f.server, f.app, f.tm)
    with pytest.raises(install.InstallationError, match="native_startup_cleanup_unconfirmed"):
        await f.server.serve()
    assert f.app.middleware_stack is None


async def test_cancelled_installer_propagates_without_serving(loop_server, monkeypatch):
    f = loop_server
    async def cancelled(*args, **kwargs): raise asyncio.CancelledError
    monkeypatch.setattr(install, "install_native_runtime", cancelled)
    install.prepare_selected_native_server(f.server, f.app, f.tm)
    with pytest.raises(asyncio.CancelledError):
        await f.server.serve()
    assert f.app.middleware_stack is None


async def test_unknown_private_installer_error_is_fixed_and_never_serves(loop_server, monkeypatch, caplog):
    f = loop_server
    async def bad(*args, **kwargs): raise RuntimeError("PRIVATE-BACKEND-DETAIL")
    monkeypatch.setattr(install, "install_native_runtime", bad)
    install.prepare_selected_native_server(f.server, f.app, f.tm)
    with pytest.raises(install.InstallationError, match="native_startup_aborted") as caught:
        await f.server.serve()
    assert "PRIVATE" not in str(caught.value) + caplog.text
    assert f.app.middleware_stack is None


def _native_records(caplog):
    return [record for record in caplog.records if record.name == "occ3_sglang.native_install"]


def test_not_selected_prepare_logs_inert_info_once(loop_server, monkeypatch, caplog):
    f = loop_server
    monkeypatch.setattr(install, "_read_startup_literal", lambda: None)
    with caplog.at_level(logging.INFO, logger="occ3_sglang.native_install"):
        result = install.prepare_selected_native_server(f.server, f.app, f.tm)
    assert not result.selected and result.code == "experimental_startup_not_selected"
    records = _native_records(caplog)
    assert len(records) == 1 and records[0].levelno == logging.INFO
    message = records[0].getMessage()
    assert re.fullmatch(r"occ3-bridge: native startup decline \(attempt=[0-9a-f]{8} "
                        r"stage=prepare code=experimental_startup_not_selected selected=false\)", message)


def test_declined_prepare_logs_fixed_receipt_code(loop_server, monkeypatch, caplog):
    f = loop_server
    def unsafe(selection):
        raise install.InstallationError("startup_selection_unsafe")
    monkeypatch.setattr(install, "_verify_startup_artifacts", unsafe)
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        result = install.prepare_selected_native_server(f.server, f.app, f.tm)
    assert result.selected and not result.prepared and result.code == "startup_selection_unsafe"
    assert getattr(f.app, install._STARTUP_ATTR)["code"] == "startup_selection_unsafe"
    records = _native_records(caplog)
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    message = records[0].getMessage()
    assert re.fullmatch(r"occ3-bridge: native startup decline \(attempt=[0-9a-f]{8} "
                        r"stage=prepare code=startup_selection_unsafe selected=True\)", message)
    assert "PRIVATE" not in message


def test_refused_prepare_logs_fixed_code_and_selected_flag(loop_server, monkeypatch, caplog):
    f = loop_server
    def broken(selection):
        raise RuntimeError("PRIVATE-GENERIC-DETAIL")
    monkeypatch.setattr(install, "_verify_startup_artifacts", broken)
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        result = install.prepare_selected_native_server(f.server, f.app, f.tm)
    assert result.selected and not result.prepared
    assert result.code == "experimental_startup_refused"
    records = _native_records(caplog)
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    message = records[0].getMessage()
    assert re.fullmatch(r"occ3-bridge: native startup decline \(attempt=[0-9a-f]{8} "
                        r"stage=prepare code=experimental_startup_refused selected=True "
                        r"detail=exception_RuntimeError\)", message)
    assert "PRIVATE" not in message and "PRIVATE" not in caplog.text


def test_refused_prepare_writes_bounded_diagnostic_file(loop_server, monkeypatch, caplog, tmp_path):
    """A generic decline leaves a read-back record; the single log line stays clean."""
    f = loop_server
    path = tmp_path / "startup-diagnostic.json"
    monkeypatch.setattr(install, "_DIAGNOSTIC_PATH", str(path))
    def broken(selection):
        raise RuntimeError("PRIVATE-GENERIC-DETAIL")
    monkeypatch.setattr(install, "_verify_startup_artifacts", broken)
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        result = install.prepare_selected_native_server(f.server, f.app, f.tm)
    assert result.code == "experimental_startup_refused"
    assert "PRIVATE" not in caplog.text
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["schema"] == "occ3.startup.diagnostic.v1"
    assert record["stage"] == "prepare" and record["selected"] is True
    assert record["exception"] == "RuntimeError" and record["exception_code"] is None
    assert [item["name"] for item in record["files"]] == [
        "native_profile.json", "native_sources.json", "_native_profile_digest.py"]
    assert "_verify_startup_artifacts" in record["traceback"]
    assert len(record["traceback"]) <= 8192


async def test_declined_install_logs_receipt_code(loop_server, monkeypatch, caplog):
    f = loop_server
    async def refuse(*args, **kwargs):
        return install.InstallationReceipt(False, "backend_not_ready")
    monkeypatch.setattr(install, "install_native_runtime", refuse)
    install.prepare_selected_native_server(f.server, f.app, f.tm)
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        assert await f.server.serve() == "native-result"
    records = _native_records(caplog)
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    message = records[0].getMessage()
    assert re.fullmatch(r"occ3-bridge: native startup decline \(attempt=[0-9a-f]{8} "
                        r"stage=install code=backend_not_ready\)", message)
    assert getattr(f.app, install._STARTUP_ATTR)["code"] == "backend_not_ready"


async def test_selected_serve_success_path_stays_silent(loop_server, caplog):
    f = loop_server
    install.prepare_selected_native_server(f.server, f.app, f.tm)
    with caplog.at_level(logging.INFO, logger="occ3_sglang.native_install"):
        assert await f.server.serve() == "native-result"
    assert _native_records(caplog) == []


def test_prepare_registers_control_plane_between_selection_and_owner_check(loop_server, monkeypatch):
    """Registration ordering: after the selection check, before the owner check."""
    f = loop_server
    order = []
    real_artifacts = install._verify_startup_artifacts
    real_server = install._verify_native_server
    monkeypatch.setattr(install, "register_control_plane",
                       lambda app, tm: order.append("register") or True)
    monkeypatch.setattr(install, "_verify_startup_artifacts",
                        lambda selection: order.append("artifacts") or real_artifacts(selection))
    monkeypatch.setattr(install, "_verify_native_server",
                        lambda s, a, t: order.append("server") or real_server(s, a, t))
    receipt = install.prepare_selected_native_server(f.server, f.app, f.tm)
    assert receipt.prepared
    assert order == ["register", "artifacts", "server"]


def test_prepare_without_selection_never_registers_the_control_plane(loop_server, monkeypatch):
    f = loop_server
    monkeypatch.setattr(install, "_read_startup_literal", lambda: None)
    calls = []
    monkeypatch.setattr(install, "register_control_plane",
                        lambda app, tm: calls.append((app, tm)) or True)
    receipt = install.prepare_selected_native_server(f.server, f.app, f.tm)
    assert not receipt.selected and calls == []
    machine = f.startup_state.current()
    assert machine.state.value == "DISABLED"
    assert machine.code == "experimental_startup_not_selected" and machine.stage == "prepare"


def test_prepare_call_order_is_pinned_by_ast():
    """Source structure: the registration call sits between selection and owner check."""
    tree = ast.parse(Path(install.__file__).read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "prepare_selected_native_server")
    statements = [node for node in function.body if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign))]
    # The try-block: selection read, not-selected early return, registration, owner check.
    body = next(node for node in statements if isinstance(node, ast.Try)).body
    names = []
    for node in body:
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                names.append(call.func.id)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                names.append(call.func.attr)
        # The not-selected early return is nested inside its guard; detect any return.
        if any(isinstance(nested, ast.Return) for nested in ast.walk(node)):
            names.append("<return>")
    assert names.index("register_control_plane") > names.index("_read_startup_literal")
    assert "<return>" in names[:names.index("register_control_plane")]  # not-selected early return
    assert names.index("register_control_plane") < names.index("getattr")  # existing-owner check


def _granian_module(monkeypatch, tmp_path, *, tamper=None):
    """Install a synthetic granian.server.embed module shaped like the pinned
    SGLang 0.5.21 import contract (``from granian.server.embed import Server``).

    The module body is written to a real file as explicit UTF-8 bytes (no
    platform newline translation) so the fd-read provenance checks operate on
    the exact written bytes on every platform; single-link file metadata is
    modeled through
    os.fstat exactly like the uvicorn tests do on Windows. The pinned expected
    digest is overridden EXPLICITLY and test-only to this synthetic file's
    actual digest; the production constant is pinned separately by
    test_granian_embed_digest_default_is_the_measured_running_value.
    """
    package = tmp_path / "granian" / "server"
    package.mkdir(parents=True)
    source = (
        "class Server:\n"
        "    def __init__(self, target, **kwargs):\n"
        "        self.target = target\n"
        "        self.kwargs = kwargs\n"
        "    async def serve(self):\n"
        "        raise AssertionError(\"never started in these tests\")\n"
        "    def stop(self):\n"
        "        return None\n"
    )
    if tamper is not None:
        source = source + tamper
    embed = package / "embed.py"
    embed.write_bytes(source.encode("utf-8"))
    module = types.ModuleType("granian")
    server_mod = types.ModuleType("granian.server")
    embed_mod = types.ModuleType("granian.server.embed")
    embed_mod.__file__ = str(embed)
    exec(compile(source, str(embed), "exec", dont_inherit=True), embed_mod.__dict__)
    server_mod.embed = embed_mod
    module.server = server_mod
    import sys as _sys
    monkeypatch.setitem(_sys.modules, "granian", module)
    monkeypatch.setitem(_sys.modules, "granian.server", server_mod)
    monkeypatch.setitem(_sys.modules, "granian.server.embed", embed_mod)
    real_fstat = os.fstat
    def single_link(fd):
        result = real_fstat(fd)
        fields = {name: getattr(result, name) for name in
            ("st_mode", "st_size", "st_dev", "st_ino", "st_mtime_ns")}
        return NS(**fields, st_nlink=1)
    monkeypatch.setattr(os, "fstat", single_link)
    monkeypatch.setattr(install, "_GRANIAN_EMBED_SHA256",
                        hashlib.sha256(embed.read_bytes()).hexdigest())
    return embed_mod, embed


def granian_server_fixture(tmp_path, monkeypatch, *, http2=True, workers=1, tamper=None):
    """Embedded-Granian-shaped server + single-tokenizer app contract."""
    embed_mod, embed = _granian_module(monkeypatch, tmp_path, tamper=tamper)
    app = FastAPI()
    args = NS(tokenizer_worker_num=workers, enable_http2=http2, enable_ssl_refresh=False)
    app.server_args = args
    app.is_single_tokenizer_mode = True
    tm = NS(server_args=args)
    server = embed_mod.Server("app-target", workers=workers)
    return server, app, tm, embed


def test_granian_verifier_accepts_only_the_embedded_single_tokenizer_http2_shape(tmp_path, monkeypatch):
    server, app, tm, embed = granian_server_fixture(tmp_path, monkeypatch)
    original = install._verify_native_server_granian(server, app, tm)
    assert original.__self__ is server and original.__func__ is type(server).serve
    assert "serve" not in vars(server) and "stop" not in vars(server)


@pytest.mark.parametrize("fault", ["wrong-class", "http2-false", "multi-worker",
                                   "ssl-refresh", "not-single-tokenizer",
                                   "instance-serve", "instance-stop", "config-attr",
                                   "wrong-app", "stack"])
def test_granian_verifier_refuses_unsupported_shapes(tmp_path, monkeypatch, fault):
    server, app, tm, embed = granian_server_fixture(tmp_path, monkeypatch)
    if fault == "wrong-class":
        server = type("Other", (), {"serve": type(server).serve, "stop": type(server).stop})()
    elif fault == "http2-false":
        tm.server_args.enable_http2 = False
    elif fault == "multi-worker":
        tm.server_args.tokenizer_worker_num = 2
    elif fault == "ssl-refresh":
        tm.server_args.enable_ssl_refresh = True
    elif fault == "not-single-tokenizer":
        app.is_single_tokenizer_mode = False
    elif fault == "instance-serve":
        server.serve = lambda: None
    elif fault == "instance-stop":
        server.stop = lambda: None
    elif fault == "config-attr":
        server.config = object()
    elif fault == "wrong-app":
        app.server_args = NS(tokenizer_worker_num=1, enable_http2=True, enable_ssl_refresh=False)
    else:
        app.middleware_stack = object()
    with pytest.raises(install.InstallationError, match="unsupported_native_startup"):
        install._verify_native_server_granian(server, app, tm)


@pytest.mark.parametrize("fault", ["symlink", "hardlink", "foreign-serve", "sync-serve",
                                   "non-module"])
def test_granian_verifier_refuses_provenance_faults(tmp_path, monkeypatch, fault):
    server, app, tm, embed = granian_server_fixture(tmp_path, monkeypatch)
    if fault == "symlink":
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink",
                            lambda p: str(p) == str(embed) or original(p))
    elif fault == "hardlink":
        real_fstat = os.fstat
        def two_links(fd):
            result = real_fstat(fd)
            fields = {name: getattr(result, name) for name in
                ("st_mode", "st_size", "st_dev", "st_ino", "st_mtime_ns")}
            return NS(**fields, st_nlink=2)
        monkeypatch.setattr(os, "fstat", two_links)
    elif fault == "foreign-serve":
        async def imposter(self):
            return None
        type(server).serve = imposter
    elif fault == "sync-serve":
        type(server).serve = lambda self: None
    else:
        import sys as _sys
        monkeypatch.setitem(_sys.modules, "granian.server.embed",
                            types.ModuleType("granian.server.embed"))
    with pytest.raises(install.InstallationError):
        install._verify_native_server_granian(server, app, tm)


def test_granian_embed_digest_default_is_the_measured_running_value():
    """The production pin is the measured running-container Granian 2.8.4
    embed.py sha256 (raw fd-read bytes); tests never change this default."""
    assert install._GRANIAN_EMBED_SHA256 == (
        "abd0b8b8cb752a0916115b3b7dd7ae6a0345ce629afb1faf2550b94adb7c001f")


def test_granian_verifier_refuses_source_digest_mismatch(tmp_path, monkeypatch):
    """A byte-level digest mismatch fails closed even when the loaded methods'
    bytecode still matches the file (digest check is independent of, and runs
    before, the function-provenance comparison)."""
    server, app, tm, embed = granian_server_fixture(
        tmp_path, monkeypatch, tamper="# unrelated byte drift\n")
    # _granian_module already overrode the pin to the TAMPERED file's digest;
    # restore the production-style expectation of the ORIGINAL content by
    # re-pinning to the untampered digest so only the file/pin pair disagrees.
    # The tamper is an exact trailing byte suffix of the written file: require
    # it once and only at the end, then strip by suffix so the removal is real
    # and never an accidental replacement of unrelated bytes.
    suffix = b"# unrelated byte drift\n"
    tampered = embed.read_bytes()
    assert tampered.count(suffix) == 1 and tampered.endswith(suffix)
    untampered = tampered.removesuffix(suffix)
    assert untampered != tampered  # tripwire: expectation must truly differ
    monkeypatch.setattr(install, "_GRANIAN_EMBED_SHA256",
                        hashlib.sha256(untampered).hexdigest())
    with pytest.raises(install.InstallationError, match="startup_server_provenance"):
        install._verify_native_server_granian(server, app, tm)


@pytest.fixture
def granian_loop_server(monkeypatch, pinned_auth, tmp_path):
    """Granian-shaped loop fixture mirroring loop_server: real class methods,
    synthetic embed module, monkeypatched verifier/installer seams."""
    from occ3_sglang import startup_state

    embed_mod, embed = _granian_module(monkeypatch, tmp_path)
    app = FastAPI()
    tm = NS(server_args=NS(api_key=KEY, tokenizer_worker_num=1,
                           enable_http2=True, enable_ssl_refresh=False))
    app.server_args = tm.server_args
    app.is_single_tokenizer_mode = True
    pinned_auth.add_api_key_middleware(app, api_key=KEY, admin_api_key=None)
    events = []

    class Server:
        async def serve(self):
            events.append(("serve", asyncio.get_running_loop()))
            assert app.middleware_stack is None
            app.middleware_stack = "native-serve-built-stack"
            return "native-granian-result"
        def stop(self):
            events.append(("stop",))
    server = Server()
    monkeypatch.setattr(install, "_read_startup_literal",
                        lambda: {"EXPERIMENTAL_STARTUP": True})
    monkeypatch.setattr(install, "_verify_startup_artifacts",
                        lambda selection: events.append(("artifacts",)))
    monkeypatch.setattr(install, "_verify_native_server_granian",
                        lambda candidate, a, t: candidate.serve)
    module_file = Path(startup_state.__file__).resolve()
    monkeypatch.setattr(install, "_verify_control_plane_sources",
                        lambda: {str(module_file): module_file.read_text(encoding="utf-8")})
    async def factory(a, t, **kwargs):
        assert a is app and t is tm and kwargs == {"experimental_opt_in": True}
        assert app.middleware_stack is None
        events.append(("install", asyncio.get_running_loop()))
        return install.InstallationReceipt(True, "installed_experimental_not_live_validated")
    monkeypatch.setattr(install, "install_native_runtime", factory)
    return NS(app=app, tm=tm, server=server, events=events, factory=factory,
              startup_state=startup_state, embed=embed, embed_mod=embed_mod)


async def test_granian_wrapper_awaits_install_on_native_loop_before_serve(granian_loop_server):
    f = granian_loop_server
    original_type_serve = type(f.server).serve
    original_type_stop = type(f.server).stop
    receipt = install.prepare_selected_native_granian_server(f.server, f.app, f.tm)
    assert receipt.prepared and not receipt.evidence()["live_proof"]
    assert [event[0] for event in f.events] == ["artifacts"]
    assert type(f.server).serve is original_type_serve
    assert type(f.server).stop is original_type_stop  # stop-hook target untouched
    result = await f.server.serve()
    assert result == "native-granian-result"
    assert [event[0] for event in f.events] == ["artifacts", "install", "serve"]
    assert f.events[1][1] is f.events[2][1] is asyncio.get_running_loop()
    assert not getattr(f.app, install._STARTUP_ATTR)["gate8_pass"]


def test_granian_absent_selection_is_inert(granian_loop_server, monkeypatch):
    f = granian_loop_server
    monkeypatch.setattr(install, "_read_startup_literal", lambda: None)
    original = f.server.serve
    receipt = install.prepare_selected_native_granian_server(f.server, f.app, f.tm)
    assert not receipt.selected and f.events == [] and f.server.serve == original
    assert "serve" not in vars(f.server)
    assert getattr(f.app, install._APP_OWNER_ATTR, None) is None


async def test_granian_duplicate_prepare_idempotent_but_run_is_one_shot(granian_loop_server):
    f = granian_loop_server
    install.prepare_selected_native_granian_server(f.server, f.app, f.tm)
    wrapper = f.server.serve
    assert install.prepare_selected_native_granian_server(f.server, f.app, f.tm).prepared
    assert f.server.serve is wrapper and len(f.events) == 1
    await wrapper()
    with pytest.raises(install.InstallationError, match="startup_owner_mismatch"):
        await wrapper()


@pytest.mark.parametrize("order", ["granian-then-uvicorn", "uvicorn-then-granian"])
async def test_cross_entry_point_double_prepare_refuses_startup_owner_mismatch(
        monkeypatch, pinned_auth, tmp_path, order):
    """One app/tokenizer pair, two entry points: the second preparation,
    in either order, is refused without wrapping a second serve method."""
    from occ3_sglang import startup_state

    embed_mod, embed = _granian_module(monkeypatch, tmp_path)
    app = FastAPI()
    tm = NS(server_args=NS(api_key=KEY, tokenizer_worker_num=1,
                           enable_http2=True, enable_ssl_refresh=False))
    app.server_args = tm.server_args
    app.is_single_tokenizer_mode = True
    pinned_auth.add_api_key_middleware(app, api_key=KEY, admin_api_key=None)
    module_file = Path(startup_state.__file__).resolve()
    monkeypatch.setattr(install, "_read_startup_literal",
                        lambda: {"EXPERIMENTAL_STARTUP": True})
    monkeypatch.setattr(install, "_verify_startup_artifacts", lambda selection: None)
    monkeypatch.setattr(install, "_verify_control_plane_sources",
                        lambda: {str(module_file): module_file.read_text(encoding="utf-8")})
    monkeypatch.setattr(install, "_verify_native_server", lambda s, a, t: s.serve)
    monkeypatch.setattr(install, "_verify_native_server_granian", lambda s, a, t: s.serve)

    class GranianServer:
        async def serve(self):
            return "granian"
        def stop(self):
            return None
    class UvicornServer:
        should_exit = False
        async def serve(self):
            return "uvicorn"

    async def factory(a, t, **kwargs):
        return install.InstallationReceipt(True, "installed_experimental_not_live_validated")
    monkeypatch.setattr(install, "install_native_runtime", factory)

    granian = GranianServer()
    uvicorn = UvicornServer()
    if order == "granian-then-uvicorn":
        first = install.prepare_selected_native_granian_server(granian, app, tm)
        assert first.prepared
        second = install.prepare_selected_native_server(uvicorn, app, tm)
    else:
        first = install.prepare_selected_native_server(uvicorn, app, tm)
        assert first.prepared
        second = install.prepare_selected_native_granian_server(granian, app, tm)
    assert not second.prepared and second.code == "startup_owner_mismatch"
    # The second entry point never installed its wrapper or recorded ownership.
    assert "serve" not in vars(uvicorn if order == "granian-then-uvicorn" else granian)
    assert getattr(app, install._APP_OWNER_ATTR) is getattr(
        granian if order == "granian-then-uvicorn" else uvicorn, install._SERVER_ATTR, None)


def test_granian_prepare_call_order_is_pinned_by_ast():
    """Source structure: the granian entry point mirrors the uvicorn ordering:
    selection read, registration, owner checks, artifacts, verifier."""
    tree = ast.parse(Path(install.__file__).read_text(encoding="utf-8"))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "prepare_selected_native_granian_server")
    statements = [node for node in function.body if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign))]
    body = next(node for node in statements if isinstance(node, ast.Try)).body
    names = []
    for node in body:
        for call in ast.walk(node):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                names.append(call.func.id)
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute):
                names.append(call.func.attr)
        if any(isinstance(nested, ast.Return) for nested in ast.walk(node)):
            names.append("<return>")
    assert names.index("register_control_plane") > names.index("_read_startup_literal")
    assert "<return>" in names[:names.index("register_control_plane")]
    assert names.index("register_control_plane") < names.index("getattr")
    assert names.index("_verify_startup_artifacts") < names.index("_verify_native_server_granian")
