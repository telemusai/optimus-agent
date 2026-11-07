"""Offline installation/auth tests, not SGLang/CUDA/TP/native semantic proof.

Auth source is the exact pinned original. Actual ASGI/FastAPI and frontend
contracts run here; model/control responses in later tests are explicit stubs.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import sys
import time
import types
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI, Request
from httpx import ASGITransport, AsyncClient
from starlette.responses import JSONResponse

from occ3_sglang import native_install as install
from occ3_sglang.attachment_contract import AttachmentError, InferenceAttachment
from occ3_sglang.latent_control_api import BrokerAuthority
from occ3_sglang.latent_types import LatentError, RuntimeProfile, SessionEnrollment

REAL_COMPONENT_FACTORY = install._construct_components


@pytest.fixture(autouse=True)
def clean_startup_state():
    """The startup singleton is process-wide; each test starts from DISABLED."""
    from occ3_sglang import startup_state

    startup_state.reset()
    yield
    startup_state.reset()

KEY = "SYNTHETIC-PRIVATE-ORIGINAL-KEY"
AUTH_SHA = "1ecd821d5fefbb196c05f33e6ee1e0cf5731cca03b27e4cce5f563ba20e6b971"
HOOK_SHA = "0144d8c3e5ee294322894f88b8ad3bd49edad15c124650e4e834d0366815dc7d"


def pinned(relative, digest):
    root = Path(os.environ.get("OCC3_SGLANG_SOURCE", str(Path(__file__).resolve().parents[4] /
        "Optimus-Assistant/profile/workspace/occ3-campaign/deps/sglang-v0.5.21")))
    file = root / "python/sglang/srt" / relative
    source = file.read_text(encoding="utf-8")
    assert hashlib.sha256(source.encode()).hexdigest() == digest
    return file, source


def module_fixture(monkeypatch, name, file, source):
    module = types.ModuleType(name)
    module.__file__ = str(file)
    monkeypatch.setitem(sys.modules, name, module)
    exec(compile(source, str(file), "exec", dont_inherit=True), module.__dict__)
    return module


@pytest.fixture
def pinned_auth(monkeypatch):
    # Only the optional JSON serializer dependency is substituted. Auth decision,
    # route matching and native ASGI middleware source run unchanged.
    import fastapi.responses
    monkeypatch.setattr(fastapi.responses, "ORJSONResponse", JSONResponse)
    file, source = pinned("utils/auth.py", AUTH_SHA)
    return module_fixture(monkeypatch, "sglang.srt.utils.auth", file, source)


def runtime_profile():
    return RuntimeProfile(runtime_epoch="fixture-runtime", runtime_generation=3,
        model_fingerprint="sha256:fixture-model", model_id="fixture-model", tile_rows=256)


def intent(**updates):
    p = runtime_profile()
    value = dict(protocol_version=1, request_id="request-1", message_id="message-1",
        connector_id="sglang-occ3", receiver_session_id="receiver-1",
        model_fingerprint=p.model_fingerprint, runtime_id=p.runtime_epoch,
        runtime_generation=p.runtime_generation, state_handle="PRIVATE-SOURCE-HANDLE",
        expires_at_epoch=int(time.time()) + 200, method=p.method)
    value.update(updates)
    return value


def authority_fixture():
    authority = BrokerAuthority(runtime_profile(), principal_id="server-owned-principal")
    p = authority.profile
    authority.remember_session(SessionEnrollment(authority.principal_digest, "receiver-1",
        p.model_fingerprint, p.runtime_epoch, p.runtime_generation,
        "trusted-family-domain", int(time.time()) + 500))
    return authority


def application(pinned_auth, *, order="marker_outer", api_key=KEY, grant_key=KEY):
    authority = authority_fixture()
    app = FastAPI()
    calls = []
    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        body = await request.json()
        if "occ3" not in body:
            return JSONResponse({"normal": body.get("messages")})
        try:
            context = await authority.authorize(request)
            context.check(InferenceAttachment.parse(body["occ3"]), body.get("model"))
            calls.append(context)
            return JSONResponse({"authorized": True, "principal_is_server_owned":
                context.principal_digest == authority.principal_digest})
        except AttachmentError as error:
            return JSONResponse({"error": error.code}, status_code=error.status)
    @app.get("/health")
    async def health(request: Request):
        return {"grant": hasattr(request.state, "_occ3_verified_broker")}
    @app.get("/unrelated")
    async def unrelated(request: Request):
        return {"grant": hasattr(request.state, "_occ3_verified_broker")}
    gate = install._BrokerGate(authority, authority.principal_id, grant_key,
        pinned_auth.decide_request_auth, pinned_auth.AuthLevel.NORMAL, 300.0)
    def normal():
        pinned_auth.add_api_key_middleware(app, api_key=api_key, admin_api_key="SYNTHETIC-ADMIN")
    if order == "marker_outer":
        normal()
        app.add_middleware(install.VerifiedBrokerMiddleware, gate=gate)
    else:
        app.add_middleware(install.VerifiedBrokerMiddleware, gate=gate)
        normal()
    return app, authority, calls


@pytest.mark.parametrize("order", ["marker_outer", "original_outer"])
async def test_verified_original_key_and_normal_chat_preserved(pinned_auth, order):
    app, authority, calls = application(pinned_auth, order=order)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://fixture") as client:
        normal = [{"role": "user", "content": "ordinary unchanged"}]
        response = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer " + KEY},
                               json={"messages": normal})
        assert response.status_code == 200 and response.json() == {"normal": normal}
        assert calls == []
        before = time.monotonic()
        result = await client.post("/v1/chat/completions", headers={"Authorization": "Bearer " + KEY},
            json={"model": "fixture-model", "occ3": intent()})
        assert result.status_code == 200 and result.json()["principal_is_server_owned"]
        assert len(calls) == 1 and before < calls[0].deadline_monotonic <= time.monotonic() + 300
        assert (await client.get("/health")).json() == {"grant": False}
        assert (await client.get("/unrelated", headers={"Authorization": "Bearer " + KEY})).json() == {"grant": False}


@pytest.mark.parametrize("header", [None, "Bearer wrong", "Basic " + KEY, "Bearer SYNTHETIC-ADMIN", "Bearer "])
@pytest.mark.parametrize("order", ["marker_outer", "original_outer"])
async def test_original_auth_denial_never_becomes_broker_grant(pinned_auth, caplog, header, order):
    app, authority, calls = application(pinned_auth, order=order)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://fixture") as client:
        headers = {} if header is None else {"Authorization": header}
        result = await client.post("/v1/chat/completions", headers=headers,
            json={"model": "fixture-model", "occ3": intent()})
        assert result.status_code == 401 and calls == []
        assert KEY not in result.text + caplog.text


@pytest.mark.parametrize("mutation", ["session", "model", "runtime", "generation", "fingerprint"])
async def test_direct_client_labels_never_autoenroll_or_override_binding(pinned_auth, mutation):
    app, authority, calls = application(pinned_auth)
    body = {"model": "fixture-model", "occ3": intent()}
    if mutation == "session": body["occ3"]["receiver_session_id"] = "not-enrolled"
    elif mutation == "model": body["model"] = "other-model"
    elif mutation == "runtime": body["occ3"]["runtime_id"] = "old-runtime"
    elif mutation == "generation": body["occ3"]["runtime_generation"] += 1
    else: body["occ3"]["model_fingerprint"] = "sha256:other"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://fixture") as client:
        result = await client.post("/v1/chat/completions", json=body,
            headers={"Authorization": "Bearer " + KEY, "X-Occ3-Principal": "forged",
                     "X-Occ3-Session": "receiver-1", "X-Occ3-Deadline": "infinite"})
        assert result.status_code == 401 and calls == []
    assert len(authority.sessions) == 1


async def test_duplicate_auth_does_not_grant_even_if_native_first_header_allows(pinned_auth):
    app, authority, calls = application(pinned_auth)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://fixture") as client:
        result = await client.post("/v1/chat/completions", json={"model": "fixture-model", "occ3": intent()},
            headers=[("Authorization", "Bearer " + KEY), ("Authorization", "Bearer wrong")])
        assert result.status_code == 401 and calls == []


async def test_no_configured_key_admin_only_options_health_never_grant(pinned_auth):
    app, authority, calls = application(pinned_auth, api_key=None, grant_key=None)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://fixture") as client:
        assert (await client.post("/v1/chat/completions", json={"messages": []})).status_code == 200
        result = await client.post("/v1/chat/completions", json={"model": "fixture-model", "occ3": intent()},
            headers={"Authorization": "Bearer SYNTHETIC-ADMIN"})
        assert result.status_code == 401 and calls == []
        assert (await client.options("/v1/chat/completions")).status_code == 405
        assert (await client.get("/health")).json() == {"grant": False}


async def test_404_unknown_route_cannot_be_counted_as_auth_or_installation_success(pinned_auth):
    app, authority, calls = application(pinned_auth)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://fixture") as client:
        result = await client.post("/occ3/v1/native/not-installed", json={},
                            headers={"Authorization": "Bearer " + KEY})
        assert result.status_code == 404 and calls == []
    assert getattr(app, install._INSTALLED_ATTR, None) is None


async def test_asgi_marker_uses_request_lifetime_and_preserves_receive_disconnect(pinned_auth):
    authority = authority_fixture()
    old = Request({"type": "http", "state": {}})
    authority.mark_verified(old, authority.principal_id, deadline_monotonic=time.monotonic() + 10)
    original_state = dict(old.scope["state"])
    observations, messages = [], [{"type": "http.request", "body": b"private-body", "more_body": False},
                                  {"type": "http.disconnect"}]
    async def receive(): return messages.pop(0)
    async def send(message): observations.append(message)
    async def app(scope, inbound, outbound):
        assert inbound is receive and outbound is send
        request = Request(scope, receive=inbound)
        assert not hasattr(request.state, "_occ3_verified_broker")
        assert await inbound() == {"type": "http.request", "body": b"private-body", "more_body": False}
        assert await inbound() == {"type": "http.disconnect"}
        observations.append(scope["state"])
    gate = install._BrokerGate(authority, authority.principal_id, KEY,
        pinned_auth.decide_request_auth, pinned_auth.AuthLevel.NORMAL)
    middleware = install.VerifiedBrokerMiddleware(app, gate=gate)
    scope = {"type": "http", "method": "POST", "path": "/v1/chat/completions",
             "headers": [(b"authorization", b"Bearer wrong")], "state": original_state}
    await middleware(scope, receive, send)
    assert observations == [{}]
    assert "_occ3_verified_broker" in original_state  # Original ASGI state was not mutated.


def test_loaded_auth_function_matches_pinned_compiled_code_and_rejects_mutation(pinned_auth):
    file, source = pinned("utils/auth.py", AUTH_SHA)
    docs = {str(file.resolve()): source}
    install._verify_function_provenance(pinned_auth.decide_request_auth, docs)
    original = pinned_auth.decide_request_auth
    def replaced(**kwargs): return NS(allowed=True)
    replaced.__module__, replaced.__qualname__ = original.__module__, original.__qualname__
    with pytest.raises(install.InstallationError):
        install._verify_function_provenance(replaced, docs)


async def test_default_install_is_inert_without_explicit_opt_in(monkeypatch):
    def forbidden(*args, **kwargs): raise AssertionError("factory must remain dormant")
    monkeypatch.setattr(install, "_construct_components", forbidden)
    result = await install.install_native_runtime(object(), object())
    assert result.evidence() == {"installed": False,
        "code": "explicit_experimental_opt_in_required", "source_ready": False,
        "receiver_ready": False, "runtime_id": None, "runtime_generation": None,
        "model_fingerprint": None, "supported": False, "live_proof": False, "gate8_pass": False}


def test_default_app_with_no_original_auth_is_refused_not_404_success():
    app = FastAPI()
    tm = NS(server_args=NS(api_key=KEY))
    with pytest.raises(install.InstallationError, match="original_auth_registration_required"):
        install._existing_auth(app, tm)


def test_existing_original_auth_must_match_current_key_without_hash_or_rotation(pinned_auth):
    app = FastAPI()
    pinned_auth.add_api_key_middleware(app, api_key=KEY, admin_api_key=None)
    assert install._existing_auth(app, NS(server_args=NS(api_key=KEY))) == KEY
    with pytest.raises(install.InstallationError, match="original_auth_mismatch"):
        install._existing_auth(app, NS(server_args=NS(api_key="different")))
    with pytest.raises(install.InstallationError, match="original_api_key_required"):
        install._existing_auth(app, NS(server_args=NS(api_key="")))


@pytest.fixture
def transaction_fixture(tmp_path, monkeypatch, pinned_auth):
    """Real adapter/ASGI/auth plus deliberately synthetic native methods/health."""
    from occ3_sglang.chat_adapter import ChatAttachmentAdapter, BufferProfile
    from occ3_sglang.attachment_contract import BackendReadiness
    from occ3_sglang.latent_control_api import create_router
    file, source = pinned("plugins/hook_registry.py", HOOK_SHA)
    hooks = module_fixture(monkeypatch, "sglang.srt.plugins.hook_registry", file, source)
    auth_file, auth_source = pinned("utils/auth.py", AUTH_SHA)
    documents = {str(auth_file.resolve()): auth_source}
    base_source = """from __future__ import annotations
class OpenAIServingBase:
    async def handle_request(self, request, raw_request):
        obj, parsed = self._convert_to_internal_request(request, raw_request)
        gen = self.tokenizer_manager.generate_request(obj, raw_request)
        try:
            reply = await gen.__anext__()
        finally:
            await gen.aclose()
        return JSONResponse(reply)
"""
    chat_source = """from __future__ import annotations
class OpenAIServingChat(OpenAIServingBase):
    def _convert_to_internal_request(self, request, raw_request=None):
        return (NS(rid='fixture-rid', input_ids=[1,2], stream=False,
                   sampling_params={'max_new_tokens': request.max_tokens}, messages=request.messages), request)
"""
    tokenizer_source = """from __future__ import annotations
class TokenizerManager:
    async def generate_request(self, obj, request=None):
        self.generations += 1
        yield {'ordinary': obj.messages, 'max_new_tokens': obj.sampling_params['max_new_tokens']}
"""
    def fixture_module(name, filename, source, **values):
        file = tmp_path / filename
        file.write_text(source, encoding="utf-8")
        module = types.ModuleType(name)
        module.__dict__.update(values)
        module.__file__ = str(file)
        monkeypatch.setitem(sys.modules, name, module)
        exec(compile(source, str(file), "exec", dont_inherit=True), module.__dict__)
        documents[str(file.resolve())] = source
        return module
    base = fixture_module("sglang.srt.entrypoints.openai.serving_base", "serving_base.py",
                          base_source, JSONResponse=JSONResponse)
    chat = fixture_module("sglang.srt.entrypoints.openai.serving_chat", "serving_chat.py",
                          chat_source, OpenAIServingBase=base.OpenAIServingBase, NS=NS)
    tokenizer = fixture_module("sglang.srt.managers.tokenizer_manager", "tokenizer_manager.py", tokenizer_source)
    tm = tokenizer.TokenizerManager()
    tm.generations = 0
    tm.server_args = NS(api_key=KEY, tokenizer_worker_num=1, device="cuda",
        tp_size=8, dp_size=1, pp_size=1, log_requests=False)
    tm.request_logger = NS(log_requests=False)
    tm.serving_chat_class = chat.OpenAIServingChat
    sg = types.ModuleType("sglang")
    sg.__version__ = "0.5.21"
    monkeypatch.setitem(sys.modules, "sglang", sg)
    app = FastAPI()
    serving = chat.OpenAIServingChat()
    serving.tokenizer_manager = tm
    @app.post("/v1/chat/completions")
    async def endpoint(raw_request: Request):
        body = await raw_request.json()
        request = NS(messages=body.get("messages", []), max_tokens=body.get("max_tokens", 4096),
                     model=body.get("model", "fixture-model"), n=1, stream=False,
                     input_ids=None, return_meta_info=False, return_hidden_states=False)
        return await serving.handle_request(request, raw_request)
    pinned_auth.add_api_key_middleware(app, api_key=KEY, admin_api_key=None)
    profile = runtime_profile()
    created = []
    refresh_gate, refresh_started = asyncio.Event(), asyncio.Event()
    controls = NS(wait=False, ready=True, fail=False, exc=None)

    class Client:
        def __init__(self):
            self.closed = False
            self.has_uncertain_control = False
            self.tm = tm
            self.calls = []
            tm._occ3_control_client = self
        async def close(self):
            self.closed = True
            if getattr(tm, "_occ3_control_client", None) is self:
                del tm._occ3_control_client
        async def call(self, operation, payload, command_id=None):
            self.calls.append((operation, payload, command_id))
            return {"state": "absent"}

    class FixtureBackend:
        def __init__(self):
            self.client = Client()
            self.closed = False
            self.attempts = {}
        async def refresh_readiness(self):
            refresh_started.set()
            if controls.wait:
                await refresh_gate.wait()
            if controls.exc is not None:
                raise controls.exc
            if controls.fail:
                raise RuntimeError(KEY)
        def readiness(self):
            return BackendReadiness(*([controls.ready] * 7))
        async def close(self):
            self.closed = True
            await self.client.close()

    async def construct(app, supplied_tm, key, buffer_profile):
        assert supplied_tm is tm and key == KEY
        authority = BrokerAuthority(profile, principal_id="server-fixture-principal")
        backend = FixtureBackend()
        adapter = ChatAttachmentAdapter(backend, authorize=authority.authorize, profile=buffer_profile)
        router = create_router(backend, authority)
        gate = install._BrokerGate(authority, authority.principal_id, key,
            pinned_auth.decide_request_auth, pinned_auth.AuthLevel.NORMAL, buffer_profile.max_turn_seconds)
        result = install._Components(profile, backend.client, backend, authority, adapter,
            router, authority.principal_id, gate, hooks.HookRegistry, hooks.HookType, hooks._wrap_fn,
            {"sglang.srt.entrypoints.openai.serving_chat.OpenAIServingChat": chat.OpenAIServingChat,
             "sglang.srt.managers.tokenizer_manager.TokenizerManager": tokenizer.TokenizerManager}, documents)
        created.append(result)
        return result
    monkeypatch.setattr(install, "_construct_components", construct)
    return NS(app=app, tm=tm, serving=serving, hooks=hooks, created=created, controls=controls,
              refresh_started=refresh_started, refresh_gate=refresh_gate,
              documents=documents, profile=profile)


async def test_install_transaction_wires_real_adapter_and_controls_but_never_gate8(transaction_fixture):
    f = transaction_fixture
    receipt = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert receipt.installed and not receipt.evidence()["gate8_pass"]
    assert not receipt.evidence()["supported"] and not receipt.evidence()["live_proof"]
    assert len(f.hooks.HookRegistry._patched) == 3 and f.tm.generations == 0
    again = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert again is receipt and len(f.created) == 1
    from httpx import ASGITransport, AsyncClient
    async with AsyncClient(transport=ASGITransport(app=f.app), base_url="http://fixture") as client:
        body = {"messages": [{"role": "user", "content": "ordinary unchanged"}], "max_tokens": 8192}
        response = await client.post("/v1/chat/completions", json=body, headers={"Authorization": "Bearer " + KEY})
        assert response.status_code == 200
        assert response.json() == {"ordinary": body["messages"], "max_new_tokens": 8192}
        assert f.tm.generations == 1
        missing = await client.post("/occ3/v1/native/publication/status",
            json={"schema": "occ3.publication.status.v1", "publication_id": "not-there"})
        assert missing.status_code == 401
        allowed = await client.post("/occ3/v1/native/publication/status",
            json={"schema": "occ3.publication.status.v1", "publication_id": "not-there"},
            headers={"Authorization": "Bearer " + KEY})
        assert allowed.status_code == 200 and allowed.json()["state"] == "absent"
        assert allowed.headers["cache-control"] == "no-store"
    assert not f.created[0].backend.closed


def app_routes(app):
    """Flatten FastAPI _IncludedRouter wrappers into their concrete routes."""
    out = []
    for route in app.router.routes:
        inner = getattr(route, "original_router", None)
        if inner is not None:
            out.extend(inner.routes)
        else:
            out.append(route)
    return out


async def test_install_mounts_identity_route_on_pinned_post_only_surface(
        transaction_fixture, monkeypatch):
    from test_latent_backend import health as backend_health
    from occ3_sglang import startup_state
    f = transaction_fixture
    startup_state.reset()
    # Phase split: the control plane (identity included) registers BEFORE install.
    module_file = Path(startup_state.__file__).resolve()
    monkeypatch.setattr(install, "_verify_control_plane_sources",
                        lambda: {str(module_file): module_file.read_text(encoding="utf-8")})
    assert install.register_control_plane(f.app, f.tm) is True
    assert startup_state.current().state.value == "CONTROL_REGISTERED"
    mounted = [getattr(route, "path", None) for route in app_routes(f.app)]
    assert mounted.count("/occ3/v1/native/runtime/identity") == 1
    assert mounted.count("/occ3/v1/status") == 1
    assert mounted.count("/occ3/v1/capabilities") == 1
    denied = await AsyncClient(transport=ASGITransport(app=f.app),
        base_url="http://fixture").post("/occ3/v1/native/runtime/identity",
        json={"schema": "occ3.runtime.identity.request.v1"},
        headers={"Authorization": "Bearer " + KEY})
    assert denied.status_code == 503 and denied.json() == {"error": {"code": "not_ready"}}
    assert denied.headers["cache-control"] == "no-store"
    # Real clients cannot reach the app before uvicorn serves; the probe above is a
    # test-only interleaving, so restore the pre-serving stack install requires.
    f.app.middleware_stack = None
    receipt = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert receipt.installed
    assert install._CONTROL_PATHS == (
        "/occ3/v1/native/sessions/enroll", "/occ3/v1/native/publish",
        "/occ3/v1/native/publication/status", "/occ3/v1/native/retire")
    components = f.created[0]
    assert ({getattr(route, "path", None) for route in components.router.routes} == set(install._CONTROL_PATHS)
            and len(components.router.routes) == len(install._CONTROL_PATHS)
            and all(getattr(route, "methods", None) == {"POST"} for route in components.router.routes))
    # The whole control surface mounts as exactly one install-owned router include.
    assert [route for route in f.app.router.routes
            if getattr(route, "original_router", None) is components.router]
    # Install never re-mounts the identity path: single ownership stays with
    # the control plane, and the pre-mounted routes are not a route conflict.
    mounted = [getattr(route, "path", None) for route in app_routes(f.app)]
    assert mounted.count("/occ3/v1/native/runtime/identity") == 1
    identity = next(route for route in app_routes(f.app)
                    if route.path == "/occ3/v1/native/runtime/identity")
    assert getattr(identity.endpoint, "__qualname__", "") == "create_control_router.<locals>.identity"
    # The FixtureBackend replies with explicit metadata only; refresh is never GPU proof.
    backend = components.backend
    backend.profile = f.profile
    async def fixture_refresh(): return backend_health(f.profile)
    backend.refresh_readiness = fixture_refresh
    async with AsyncClient(transport=ASGITransport(app=f.app), base_url="http://fixture") as client:
        no_key = await client.post("/occ3/v1/native/runtime/identity",
            json={"schema": "occ3.runtime.identity.request.v1"})
        assert no_key.status_code == 401
        allowed = await client.post("/occ3/v1/native/runtime/identity",
            json={"schema": "occ3.runtime.identity.request.v1"},
            headers={"Authorization": "Bearer " + KEY})
        assert allowed.status_code == 200, allowed.text
        assert allowed.json()["runtime_id"] == f.profile.runtime_epoch
        assert allowed.json()["runtime_generation"] == f.profile.runtime_generation
        assert allowed.headers["cache-control"] == "no-store"
    startup_state.reset()


@pytest.mark.parametrize("phase", ["router", "middleware", "hook"])
async def test_partial_install_rollback_retry_keeps_unrelated_state(transaction_fixture, monkeypatch, caplog, phase):
    f = transaction_fixture
    routes_before = list(f.app.router.routes)
    middleware_before = list(f.app.user_middleware)
    unrelated = lambda: None
    f.hooks.HookRegistry._hooks["unrelated.target"].append((f.hooks.HookType.AFTER, unrelated, None))
    f.hooks.HookRegistry._patched.add("unrelated.target")
    restore = None
    if phase == "router":
        original = f.app.include_router
        def broken(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError(KEY)
        monkeypatch.setattr(f.app, "include_router", broken)
        restore = lambda: monkeypatch.setattr(f.app, "include_router", original)
    elif phase == "middleware":
        original = f.app.add_middleware
        def broken(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError(KEY)
        monkeypatch.setattr(f.app, "add_middleware", broken)
        restore = lambda: monkeypatch.setattr(f.app, "add_middleware", original)
    else:
        original = f.hooks.HookRegistry.register
        calls = [0]
        def broken(target, hook, kind):
            original(target, hook, kind)
            calls[0] += 1
            if calls[0] == 2:
                raise RuntimeError(KEY)
        monkeypatch.setattr(f.hooks.HookRegistry, "register", broken)
        restore = lambda: monkeypatch.setattr(f.hooks.HookRegistry, "register", original)
    result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert not result.installed and result.code == "installation_unavailable"
    assert f.app.router.routes == routes_before and f.app.user_middleware == middleware_before
    assert set(f.hooks.HookRegistry._hooks) == {"unrelated.target"}
    assert f.hooks.HookRegistry._patched == {"unrelated.target"}
    assert getattr(f.tm, "_occ3_control_client", None) is None and f.created[0].backend.closed
    assert "handle_request" not in vars(type(f.serving))  # Inherited method was restored.
    assert KEY not in str(result.evidence()) + caplog.text
    restore()
    retried = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert retried.installed and len(f.hooks.HookRegistry._patched) == 4


@pytest.mark.parametrize("fault,code", [("not_ready", "backend_not_ready"),
    ("logging", "unsupported_frontend_profile"), ("late", "pre_serving_install_required"),
    ("occupied", "hook_target_occupied"), ("control_client", "control_client_occupied"),
    ("bad_runtime", "unsupported_runtime")])
async def test_install_refuses_unknown_or_occupied_profiles_without_generation(transaction_fixture, fault, code):
    f = transaction_fixture
    routes_before = list(f.app.router.routes)
    if fault == "not_ready": f.controls.ready = False
    elif fault == "logging": f.tm.request_logger.log_requests = True
    elif fault == "late": f.app.middleware_stack = object()
    elif fault == "occupied": f.hooks.HookRegistry._patched.add(install._EXPECTED_HOOKS[0][0])
    elif fault == "control_client": f.tm._occ3_control_client = object()
    else: sys.modules["sglang"].__version__ = "other"
    result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert not result.installed and result.code == code
    assert f.app.router.routes == routes_before and f.tm.generations == 0


async def test_install_conflicting_control_route_is_not_hidden_by_404_or_duplicate_mount(transaction_fixture):
    f = transaction_fixture
    @f.app.post("/occ3/{rest:path}")
    async def catchall(): return {"unrelated": True}
    result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert not result.installed and result.code == "native_route_conflict"
    assert f.created == [] and f.tm.generations == 0


async def test_install_cancel_during_readiness_rolls_back_and_releases_transaction_lock(transaction_fixture):
    f = transaction_fixture
    f.controls.wait = True
    task = asyncio.create_task(install.install_native_runtime(f.app, f.tm, experimental_opt_in=True))
    await asyncio.wait_for(f.refresh_started.wait(), .5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert getattr(f.tm, "_occ3_control_client", None) is None
    assert not f.hooks.HookRegistry._patched and not install._install_lock.locked()
    assert f.created[0].backend.closed
    f.controls.wait = False
    assert (await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)).installed


async def test_install_failure_uses_fixed_error_and_no_secret_logging(transaction_fixture, caplog):
    f = transaction_fixture
    f.controls.fail = True
    result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert result.code == "installation_unavailable" and not result.installed
    assert KEY not in str(result.evidence()) + caplog.text
    assert f.created[0].backend.closed and not install._install_lock.locked()


async def test_mounted_install_logs_success_once(transaction_fixture, caplog):
    f = transaction_fixture
    with caplog.at_level(logging.INFO, logger="occ3_sglang.native_install"):
        first = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
        again = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert first.installed and again is first
    records = [record for record in caplog.records if record.name == "occ3_sglang.native_install"]
    mounted = [record for record in records if "native runtime mounted" in record.getMessage()]
    assert len(mounted) == 1 and mounted[0].levelno == logging.INFO
    assert re.fullmatch(r"occ3-bridge: native runtime mounted \(attempt=[0-9a-f]{8} "
                       r"stage=install code=installed_experimental_not_live_validated\)",
                       mounted[0].getMessage())
    assert not [record for record in records if record.levelno >= logging.ERROR]


async def test_declined_install_logs_final_return_code(transaction_fixture, caplog):
    from occ3_sglang import startup_state
    f = transaction_fixture
    f.controls.ready = False
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert not result.installed and result.code == "backend_not_ready"
    records = [record for record in caplog.records if record.name == "occ3_sglang.native_install"]
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    message = records[0].getMessage()
    assert re.fullmatch(r"occ3-bridge: native installation decline \(attempt=[0-9a-f]{8} "
                       r"stage=install code=backend_not_ready\)", message)
    assert KEY not in caplog.text
    # The machine records the same finalize site: exact code, reached stage and
    # the attempt marker shared with the C30 line above.
    machine = startup_state.current()
    assert machine.state.value == "DECLINED" and machine.code == "backend_not_ready"
    assert machine.stage == "install" and machine.attempt == re.search(
        r"attempt=([0-9a-f]{8})", message).group(1)
    assert machine.experimental_selected and machine.preflight_reached
    assert machine.backend_constructed and machine.hooks_prepared
    assert machine.worker_readiness_reached and not machine.route_mount_attempted
    assert machine.rollback_completed is True  # verified clean decline, nothing pending


async def test_declined_install_logs_sanitised_timeout_detail(transaction_fixture, caplog):
    f = transaction_fixture
    f.controls.exc = asyncio.TimeoutError()
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert result.code == "installation_unavailable" and not result.installed
    records = [record for record in caplog.records if record.name == "occ3_sglang.native_install"]
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    assert re.fullmatch(r"occ3-bridge: native installation decline \(attempt=[0-9a-f]{8} "
                       r"stage=install code=installation_unavailable detail=control_timeout\)",
                       records[0].getMessage())


async def test_declined_install_logs_sanitised_latent_detail(transaction_fixture, caplog):
    f = transaction_fixture
    f.controls.exc = LatentError("profile")
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert result.code == "installation_unavailable" and not result.installed
    records = [record for record in caplog.records if record.name == "occ3_sglang.native_install"]
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    assert re.fullmatch(r"occ3-bridge: native installation decline \(attempt=[0-9a-f]{8} "
                       r"stage=install code=installation_unavailable detail=latent_profile\)",
                       records[0].getMessage())


async def test_declined_install_generic_detail_keeps_taxonomy_and_hides_text(transaction_fixture, caplog):
    f = transaction_fixture
    f.controls.fail = True
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert result.code == "installation_unavailable" and not result.installed
    assert result.evidence() == {"code": "installation_unavailable", "installed": False,
                                "live_proof": False, "gate8_pass": False, "supported": False,
                                "source_ready": False, "receiver_ready": False,
                                "runtime_id": None, "runtime_generation": None,
                                "model_fingerprint": None}
    records = [record for record in caplog.records if record.name == "occ3_sglang.native_install"]
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    assert re.fullmatch(r"occ3-bridge: native installation decline \(attempt=[0-9a-f]{8} "
                       r"stage=install code=installation_unavailable detail=exception_RuntimeError\)",
                       records[0].getMessage())
    assert KEY not in caplog.text


async def test_duplicate_install_rejects_lost_hook_owner_without_overwriting_unknown_wrapper(transaction_fixture):
    f = transaction_fixture
    assert (await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)).installed
    unknown = lambda *args: None
    type(f.serving).handle_request = unknown
    result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert result.code == "installation_owner_mismatch" and not result.installed
    assert type(f.serving).handle_request is unknown and not f.created[0].backend.closed


async def test_plugin_exports_explicit_installer_but_ordinary_load_never_calls_it(monkeypatch):
    from occ3_sglang import plugin
    calls = []
    async def fake(app, tm, **kwargs):
        calls.append((app, tm, kwargs))
        return install.InstallationReceipt(False, "fixture")
    monkeypatch.setattr(install, "install_native_runtime", fake)
    assert calls == []
    app, tm = object(), object()
    result = await plugin.install_native(app, tm)
    assert not result.installed and calls == [(app, tm,
        {"experimental_opt_in": False, "buffer_profile": None})]


@pytest.fixture
def attested_tree(tmp_path):
    roots = {name: tmp_path / name for name in ("sglang_srt", "occ3_sglang")}
    files = []
    for root, paths in (("sglang_srt", install._REQUIRED_SGLANG), ("occ3_sglang", install._REQUIRED_PLUGIN)):
        for relative in sorted(paths):
            path = roots[root] / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            data = b"# explicit synthetic file identity fixture, not executable overlay\n"
            path.write_bytes(data)
            files.append({"root": root, "path": relative, "sha256": hashlib.sha256(data).hexdigest()})
    value = {"schema": "optimus.occ3.native-sources.v1", "source_revision": install._SOURCE_REVISION,
             "sglang_version": "0.5.21", "manifests": [{"name": "fixture", "sha256": "a" * 64}],
             "files": files}
    return roots, value


@pytest.mark.parametrize("module", ["startup_state.py", "turn_diagnostics.py"])
def test_runtime_module_is_a_required_attested_plugin_member(attested_tree, module):
    roots, value = attested_tree
    assert module in install._REQUIRED_PLUGIN
    value["files"] = [item for item in value["files"]
                      if not (item["root"] == "occ3_sglang" and item["path"] == module)]
    with pytest.raises(install.InstallationError, match="source_attestation_incomplete"):
        install._verify_installed_sources(value, roots)


def test_source_attestation_reads_exact_installed_bytes_without_executing(attested_tree):
    roots, value = attested_tree
    documents = install._verify_installed_sources(value, roots)
    assert len(documents) == len(value["files"])
    assert all("synthetic" in text for text in documents.values())


@pytest.mark.parametrize("fault", ["hash", "missing", "duplicate", "traversal", "root", "oversize", "revision"])
def test_source_attestation_refuses_mismatch_incomplete_escape_or_bounds(attested_tree, monkeypatch, fault):
    roots, value = attested_tree
    first = value["files"][0]
    path = roots[first["root"]] / first["path"]
    if fault == "hash": path.write_bytes(b"modified")
    elif fault == "missing": value["files"].pop()
    elif fault == "duplicate": value["files"].append(dict(first))
    elif fault == "traversal": first["path"] = "../private.py"
    elif fault == "root": first["root"] = "arbitrary"
    elif fault == "oversize": monkeypatch.setattr(install, "_MAX_SOURCE_FILE", 1)
    else: value["source_revision"] = "unreviewed"
    with pytest.raises(install.InstallationError):
        install._verify_installed_sources(value, roots)


def test_source_attestation_rejects_symlink_boundary_without_following(attested_tree, monkeypatch):
    roots, value = attested_tree
    original = Path.is_symlink
    target = roots[value["files"][0]["root"]] / value["files"][0]["path"]
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == target or original(path))
    with pytest.raises(install.InstallationError, match="source_symlink"):
        install._verify_installed_sources(value, roots)


def test_exact_native_auth_middleware_method_provenance_is_verified(pinned_auth):
    app = FastAPI()
    pinned_auth.add_api_key_middleware(app, api_key=KEY, admin_api_key=None)
    file, source = pinned("utils/auth.py", AUTH_SHA)
    install._verify_function_provenance(app.user_middleware[0].cls.__call__, {str(file.resolve()): source})


async def test_source_and_profile_missing_factory_cannot_install_anything(transaction_fixture, monkeypatch):
    f = transaction_fixture
    async def unavailable(*args):
        raise FileNotFoundError("SYNTHETIC-PRIVATE-PROFILE-PATH")
    monkeypatch.setattr(install, "_construct_components", unavailable)
    before = list(f.app.router.routes)
    result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert not result.installed and result.code == "installation_unavailable"
    assert f.app.router.routes == before and not f.hooks.HookRegistry._patched
    assert f.tm.generations == 0 and "SYNTHETIC-PRIVATE" not in str(result.evidence())


@pytest.mark.parametrize("fault", ["uncertain", "missing-contract", "still-attached", "not-closed"])
async def test_cleanup_requires_actual_settlement_not_empty_local_pending(transaction_fixture, fault):
    f = transaction_fixture
    components = await install._construct_components(f.app, f.tm, KEY,
        __import__("occ3_sglang.chat_adapter", fromlist=["BufferProfile"]).BufferProfile())
    if fault == "uncertain": components.client.has_uncertain_control = True
    elif fault == "missing-contract": del components.client.has_uncertain_control
    else:
        async def close():
            components.backend.closed = fault != "not-closed"
            components.client.closed = True
        components.backend.close = close
    assert not await install._close_components(components)


async def test_real_frontend_factory_assembles_controls_auth_adapter_with_explicit_native_stubs(
        transaction_fixture, monkeypatch, tmp_path, pinned_auth):
    """Real A frontend classes/factory; declared IPC/engine are explicit stubs.

    Installed-source IO/provenance is separately tested. This test is assembly,
    not a live worker readiness or semantic proof.
    """
    from dataclasses import make_dataclass
    from occ3_sglang import native_profile, latent_worker, latent_backend, latent_control_api
    from occ3_sglang.chat_adapter import BufferProfile
    f = transaction_fixture
    docs = dict(f.documents)
    for relative, sha in (("utils/auth.py", AUTH_SHA), ("plugins/hook_registry.py", HOOK_SHA)):
        path, source = pinned(relative, sha)
        docs[str(path.resolve())] = source
    path = Path(latent_control_api.__file__)
    docs[str(path.resolve())] = path.read_text(encoding="utf-8")
    monkeypatch.setattr(native_profile, "load_source_attestation", lambda: {"explicit-fixture": True})
    monkeypatch.setattr(install, "_verify_installed_sources", lambda value, roots: docs)
    monkeypatch.setattr(latent_worker, "load_runtime_profile", lambda: f.profile)
    srt = types.ModuleType("sglang.srt")
    (tmp_path / "srt").mkdir()
    srt.__path__ = [str(tmp_path / "srt")]
    srt.__file__ = None  # Exact pinned srt is a namespace package.
    monkeypatch.setitem(sys.modules, "sglang.srt", srt)
    io = types.ModuleType("sglang.srt.managers.io_struct")
    io.GenerateReqInput = make_dataclass("GenerateReqInput", [("occ3_worker_descriptor", object, None)])
    io.TokenizedGenerateReqInput = type("TokenizedGenerateReqInput", (),
        {"__struct_fields__": ("rid", "occ3_worker_descriptor")})
    class Control:
        __struct_fields__ = ("rid", "payload")
        def __init__(self, *, rid, payload): self.rid, self.payload = rid, payload
    io.Occ3ControlReq = Control
    io.Occ3ControlOutput = Control
    monkeypatch.setitem(sys.modules, "sglang.srt.managers.io_struct", io)
    f.tm.auto_create_handle_loop = lambda: None
    requests = []
    async def dispatch(request):
        requests.append(request)
        p = f.profile
        health = {"runtime_epoch": p.runtime_epoch, "runtime_generation": p.runtime_generation,
            "model_fingerprint": p.model_fingerprint, "method_id": p.method, "hidden_dim": p.hidden_dim,
            "layer_index": p.layer_index, "ready_ranks": p.expected_ranks, "expected_ranks": p.expected_ranks,
            "profile_digest": p.binding_digest, "source_capture": True, "continuous_prefix": True,
            "terminal_journal": True, "max_capture_bytes": 128 * 1024 * 1024}
        latent_backend.deliver_control(f.tm, {"schema": "occ3.control.result.v1",
            "command_id": request.rid, "ok": True, "result": health, "error": ""})
    f.tm._async_dispatch_to_scheduler = dispatch
    before_torch = sys.modules.get("torch")
    components = await REAL_COMPONENT_FACTORY(f.app, f.tm, KEY, BufferProfile())
    assert type(components.backend) is latent_backend.LatentBackend
    assert type(components.authority) is latent_control_api.BrokerAuthority
    assert components.gate.api_key == KEY and not components.backend.readiness().receiver_ready
    assert f.tm.generations == 0 and not requests
    assert sys.modules.get("torch") is before_torch
    pending = install._prepare_hooks(components)
    assert len(pending) == 3 and not f.hooks.HookRegistry._patched
    await components.backend.refresh_readiness()
    assert len(requests) == 1 and requests[0].payload["operation"] == "health"
    assert components.backend.readiness().source_ready  # Explicit fake engine health only.
    assert not components.client.has_uncertain_control
    assert await install._close_components(components)
    assert not hasattr(f.tm, "_occ3_control_client") and f.tm.generations == 0


@pytest.mark.parametrize("fault", ["generate", "tokenized", "request", "output"])
def test_overlay_check_distinguishes_dataclass_from_msgspec_fields(fault):
    from dataclasses import make_dataclass
    generate = make_dataclass("GenerateReqInput", [("occ3_worker_descriptor", object, None)])
    tokenized = NS(__struct_fields__=("rid", "occ3_worker_descriptor"))
    req = NS(__struct_fields__=("rid", "payload"))
    out = NS(__struct_fields__=("rid", "payload"))
    install._verify_descriptor_overlay(generate, tokenized, req, out)
    if fault == "generate": generate = object()
    elif fault == "tokenized": tokenized.__struct_fields__ = ("rid",)
    elif fault == "request": req.__struct_fields__ = ("payload",)
    else: out.__struct_fields__ = ("rid",)
    with pytest.raises(install.InstallationError, match="descriptor_overlay_unavailable"):
        install._verify_descriptor_overlay(generate, tokenized, req, out)


def test_namespace_srt_source_root_is_single_explicit_location(tmp_path):
    root = tmp_path / "srt"
    root.mkdir()
    package = NS(__path__=[str(root)], __file__=None)
    assert install._sglang_srt_root(package) == root
    package.__file__ = str(root / "__init__.py")
    assert install._sglang_srt_root(package) == root


@pytest.mark.parametrize("fault", ["absent", "multiple", "missing", "conflict", "symlink"])
def test_namespace_source_root_rejects_ambiguous_or_unsafe_location(tmp_path, monkeypatch, fault):
    root = tmp_path / "srt"
    root.mkdir()
    package = NS(__path__=[str(root)], __file__=None)
    if fault == "absent": del package.__path__
    elif fault == "multiple": package.__path__.append(str(tmp_path))
    elif fault == "missing": package.__path__ = [str(root / "missing")]
    elif fault == "conflict": package.__file__ = str(tmp_path / "elsewhere/__init__.py")
    else:
        original = Path.is_symlink
        monkeypatch.setattr(Path, "is_symlink", lambda value: value == root or original(value))
    with pytest.raises(install.InstallationError, match="source_root_unavailable"):
        install._sglang_srt_root(package)


@pytest.mark.parametrize("change", ["hook", "hook-code", "auth-code", "registry-entry", "registry-patched", "registry-container",
    "catchall", "mount", "route-inplace", "middleware", "auth-key", "client", "app-marker",
    "openapi", "route-list", "server-args", "built-stack"])
async def test_readiness_await_cannot_overwrite_or_adopt_foreign_ownership(transaction_fixture, change):
    import re
    from starlette.routing import Mount, Route
    f = transaction_fixture
    f.controls.wait = True
    task = asyncio.create_task(install.install_native_runtime(f.app, f.tm, experimental_opt_in=True))
    await asyncio.wait_for(f.refresh_started.wait(), 1)
    components = f.created[0]
    target = install._EXPECTED_HOOKS[0][0]
    foreign = object()
    async def wrapper(*args, **kwargs): return "foreign-wrapper"
    async def endpoint(request): return JSONResponse({"foreign": True})
    if change == "hook":
        type(f.serving).handle_request = wrapper
        preserved = lambda: type(f.serving).handle_request is wrapper
    elif change in ("hook-code", "auth-code"):
        function = (type(f.serving).handle_request if change == "hook-code"
                    else f.app.user_middleware[0].cls.__call__)
        foreign = function.__code__.replace()  # Even equivalent new code is unknown ownership.
        function.__code__ = foreign
        preserved = lambda: function.__code__ is foreign
    elif change == "registry-entry":
        entry = (f.hooks.HookType.AROUND, wrapper)
        f.hooks.HookRegistry._hooks[target] = [entry]
        preserved = lambda: f.hooks.HookRegistry._hooks[target] == [entry]
    elif change == "registry-patched":
        f.hooks.HookRegistry._patched.add(target)
        preserved = lambda: target in f.hooks.HookRegistry._patched
    elif change == "registry-container":
        foreign = dict(f.hooks.HookRegistry._hooks)
        f.hooks.HookRegistry._hooks = foreign
        preserved = lambda: f.hooks.HookRegistry._hooks is foreign
    elif change in ("catchall", "mount"):
        foreign = (Route("/occ3/{rest:path}", endpoint, methods=["POST"]) if change == "catchall"
                   else Mount("/occ3", app=FastAPI()))
        f.app.router.routes.insert(0, foreign)
        preserved = lambda: f.app.router.routes[0] is foreign
    elif change == "route-inplace":
        foreign = f.app.router.routes[-1]
        foreign.path_regex = re.compile(r"^/occ3/.*$")
        foreign.methods = {"POST"}
        preserved = lambda: foreign.path_regex.pattern == r"^/occ3/.*$" and foreign.methods == {"POST"}
    elif change == "middleware":
        class OtherMiddleware:
            def __init__(self, app): self.app = app
            async def __call__(self, scope, receive, send): await self.app(scope, receive, send)
        f.app.add_middleware(OtherMiddleware)
        foreign = f.app.user_middleware[0]
        preserved = lambda: f.app.user_middleware[0] is foreign
    elif change == "auth-key":
        auth = f.app.user_middleware[0]
        auth.kwargs["api_key"] = "DIFFERENT-PRIVATE-KEY"
        preserved = lambda: auth.kwargs["api_key"] == "DIFFERENT-PRIVATE-KEY"
    elif change == "client":
        f.tm._occ3_control_client = foreign
        preserved = lambda: f.tm._occ3_control_client is foreign
    elif change == "app-marker":
        setattr(f.app, install._INSTALLED_ATTR, foreign)
        preserved = lambda: getattr(f.app, install._INSTALLED_ATTR) is foreign
    elif change == "openapi":
        f.app.openapi_schema = foreign
        preserved = lambda: f.app.openapi_schema is foreign
    elif change == "route-list":
        foreign = list(f.app.router.routes)
        f.app.router.routes = foreign
        preserved = lambda: f.app.router.routes is foreign
    elif change == "server-args":
        foreign = NS(**vars(f.tm.server_args))
        f.tm.server_args = foreign
        preserved = lambda: f.tm.server_args is foreign
    else:
        f.app.middleware_stack = foreign
        preserved = lambda: f.app.middleware_stack is foreign
    f.refresh_gate.set()
    result = await asyncio.wait_for(task, 2)
    assert not result.installed and result.code == "installation_cleanup_unconfirmed"
    assert preserved() and components.backend.closed and components.client.closed
    assert f.tm.generations == 0
    assert not any(getattr(r, "original_router", None) is components.router for r in f.app.router.routes)
    assert not any(m.cls is install.VerifiedBrokerMiddleware for m in f.app.user_middleware)
    assert "DIFFERENT-PRIVATE-KEY" not in str(result.evidence())


async def test_unrelated_registry_entry_during_readiness_is_preserved_and_not_adopted(transaction_fixture):
    f = transaction_fixture
    f.controls.wait = True
    task = asyncio.create_task(install.install_native_runtime(f.app, f.tm, experimental_opt_in=True))
    await asyncio.wait_for(f.refresh_started.wait(), 1)
    unrelated = (object(), object())
    f.hooks.HookRegistry._hooks["unrelated.explicit.target"] = [unrelated]
    f.hooks.HookRegistry._patched.add("unrelated.explicit.target")
    f.refresh_gate.set()
    result = await asyncio.wait_for(task, 2)
    assert result.installed
    assert f.hooks.HookRegistry._hooks["unrelated.explicit.target"] == [unrelated]
    assert "unrelated.explicit.target" in f.hooks.HookRegistry._patched


async def test_cleanup_await_revalidates_shared_state_before_safe_refusal(transaction_fixture):
    f = transaction_fixture
    f.controls.wait = True
    f.controls.ready = False
    task = asyncio.create_task(install.install_native_runtime(f.app, f.tm, experimental_opt_in=True))
    await asyncio.wait_for(f.refresh_started.wait(), 1)
    components = f.created[0]
    close_started, close_release = asyncio.Event(), asyncio.Event()
    original_close = components.backend.close
    async def delayed_close():
        close_started.set()
        await close_release.wait()
        await original_close()
    components.backend.close = delayed_close
    f.refresh_gate.set()
    await asyncio.wait_for(close_started.wait(), 1)
    foreign = object()
    f.app.openapi_schema = foreign
    close_release.set()
    result = await asyncio.wait_for(task, 2)
    assert not result.installed and result.code == "installation_cleanup_unconfirmed"
    assert f.app.openapi_schema is foreign and components.client.closed


async def test_cancelled_readiness_late_reply_never_installs(transaction_fixture):
    f = transaction_fixture
    f.controls.wait = True
    task = asyncio.create_task(install.install_native_runtime(f.app, f.tm, experimental_opt_in=True))
    await asyncio.wait_for(f.refresh_started.wait(), 1)
    before_routes = tuple(f.app.router.routes)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    f.refresh_gate.set()
    assert tuple(f.app.router.routes) == before_routes
    assert not f.hooks.HookRegistry._patched
    assert not hasattr(f.app, install._INSTALLED_ATTR)
    assert f.created[0].client.closed


async def test_selected_startup_never_serves_after_real_ownership_race(transaction_fixture, monkeypatch):
    f = transaction_fixture
    f.controls.wait = True
    calls = []
    class Server:
        async def serve(self, **kwargs): calls.append(kwargs)
    server = Server()
    monkeypatch.setattr(install, "_read_startup_literal", lambda: {"EXPERIMENTAL_STARTUP": True})
    monkeypatch.setattr(install, "_verify_startup_artifacts", lambda value: None)
    monkeypatch.setattr(install, "_verify_native_server", lambda s, a, t: s.serve)
    install.prepare_selected_native_server(server, f.app, f.tm)
    task = asyncio.create_task(server.serve(sockets=None))
    await asyncio.wait_for(f.refresh_started.wait(), 1)
    async def foreign(*args, **kwargs): return None
    type(f.serving).handle_request = foreign
    f.refresh_gate.set()
    with pytest.raises(install.InstallationError, match="native_startup_cleanup_unconfirmed"):
        await task
    assert not calls and type(f.serving).handle_request is foreign


async def test_cancellation_resistant_late_health_cannot_commit(transaction_fixture, monkeypatch):
    f = transaction_fixture
    construct = install._construct_components
    suppressed = asyncio.Event()
    async def make(*args):
        components = await construct(*args)
        async def resistant_health():
            f.refresh_started.set()
            try:
                await f.refresh_gate.wait()
            except asyncio.CancelledError:
                suppressed.set()
                await f.refresh_gate.wait()
        components.backend.refresh_readiness = resistant_health
        return components
    monkeypatch.setattr(install, "_construct_components", make)
    task = asyncio.create_task(install.install_native_runtime(f.app, f.tm, experimental_opt_in=True))
    await asyncio.wait_for(f.refresh_started.wait(), 1)
    task.cancel()
    await asyncio.wait_for(suppressed.wait(), 1)
    f.refresh_gate.set()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, 2)
    assert not f.hooks.HookRegistry._patched and f.created[0].client.closed
    assert not hasattr(f.app, install._INSTALLED_ATTR)


async def test_actual_A_control_uncertainty_survives_local_close(monkeypatch):
    from occ3_sglang.latent_backend import ControlClient, LatentBackend
    io = types.ModuleType("sglang.srt.managers.io_struct")
    class Occ3ControlReq:
        def __init__(self, *, rid, payload): self.rid, self.payload = rid, payload
    io.Occ3ControlReq = Occ3ControlReq
    monkeypatch.setitem(sys.modules, io.__name__, io)
    async def unknown_dispatch(request):
        raise RuntimeError("SYNTHETIC-UNKNOWN-DISPATCH")
    tm = NS(auto_create_handle_loop=lambda: None, _async_dispatch_to_scheduler=unknown_dispatch)
    client = ControlClient(tm, timeout_seconds=1)
    p = runtime_profile()
    backend = LatentBackend(client, p, BrokerAuthority(p, principal_id="fixture-principal"))
    with pytest.raises(RuntimeError, match="SYNTHETIC-UNKNOWN-DISPATCH"):
        await client.call("health", {})
    assert await install._close_components(NS(client=client, backend=backend)) is False
    assert client.closed and client.has_uncertain_control and client.pending == {}
    assert tm._occ3_control_client is client


async def test_frontend_preflight_accepts_autodetected_cuda(transaction_fixture, monkeypatch):
    """sglang 0.5.21 auto-detect leaves server_args.device None; the runtime
    device confirms cuda (TS5 root cause: the literal attribute declined
    every auto-detect launch)."""
    f = transaction_fixture
    monkeypatch.setattr(f.tm.server_args, "device", None)
    monkeypatch.setattr(install, "_resolve_runtime_device", lambda: "cuda")
    members = dict(install._frontend_members(f.app, f.tm))
    assert members["device"] is False
    assert install._runtime_preflight(f.app, f.tm) == KEY


async def test_frontend_preflight_fails_closed_without_runtime_device(transaction_fixture, monkeypatch):
    f = transaction_fixture
    monkeypatch.setattr(f.tm.server_args, "device", None)
    monkeypatch.setattr(install, "_resolve_runtime_device", lambda: None)
    members = dict(install._frontend_members(f.app, f.tm))
    assert members["device"] is True


async def test_frontend_preflight_declines_explicit_non_cuda(transaction_fixture, monkeypatch):
    f = transaction_fixture
    monkeypatch.setattr(f.tm.server_args, "device", "xpu")
    monkeypatch.setattr(install, "_resolve_runtime_device", lambda: "cuda")
    members = dict(install._frontend_members(f.app, f.tm))
    assert members["device"] is True


async def test_frontend_decline_names_every_tripping_member(transaction_fixture, monkeypatch, caplog):
    f = transaction_fixture
    monkeypatch.setattr(f.tm.server_args, "device", None)
    monkeypatch.setattr(f.tm.server_args, "tp_size", 4)
    f.tm.request_logger.log_requests = True
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert not result.installed and result.code == "unsupported_frontend_profile"
    message = [r for r in caplog.records if "native installation decline" in r.getMessage()][0]
    assert "detail=frontend_members=device,tp_size,request_logger" in message.getMessage()
    assert KEY not in caplog.text


async def test_frontend_decline_token_is_bounded_and_sanitised():
    error = install.InstallationError("unsupported_frontend_profile")
    error.frontend_members = "a" * 500
    token = install._frontend_detail(error)
    assert token == "frontend_members=" + "a" * 111 and len(token) == 128
    assert install._frontend_detail(install.InstallationError("other")) is None
    assert install._frontend_detail(RuntimeError("x")) is None


async def test_frontend_preflight_logger_and_root_path_members(transaction_fixture):
    f = transaction_fixture
    f.tm.request_logger.log_requests = True
    f.app.root_path = "/proxy"
    members = dict(install._frontend_members(f.app, f.tm))
    assert members["request_logger"] is True and members["root_path"] is True


##############################
# OCC3_MAX_TURN_SECONDS: per-boot native turn admission window (TS13 live regression)

async def test_max_turn_window_env_absent_keeps_default_300(transaction_fixture, monkeypatch):
    """The default admission window stays 300s when no override is declared.
    This is the value that capped the TS13 armed turn: the descriptor declared
    deadline_seconds=540 but the identity admission window (buffer profile
    max_turn_seconds) folded ctx.deadline to request-start+300s and the live
    armed decode died with deadline_exceeded/504 at ~301s mid-generation."""
    monkeypatch.delenv("OCC3_MAX_TURN_SECONDS", raising=False)
    f = transaction_fixture
    receipt = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert receipt.installed
    installation = getattr(f.app, install._INSTALLED_ATTR)
    assert installation.adapter.profile.max_turn_seconds == 300.0


async def test_max_turn_window_env_widens_admission_to_declared_boot_window(
        transaction_fixture, monkeypatch):
    """Red-first, TS13 live regression: deployments serving armed turns opt in
    per-boot with OCC3_MAX_TURN_SECONDS. The declared window must reach the
    installed adapter profile (the same value _construct_components hands the
    broker gate as its admission window), so a 540s armed descriptor deadline
    fits inside the identity admission instead of dying mid-decode."""
    monkeypatch.setenv("OCC3_MAX_TURN_SECONDS", "720")
    f = transaction_fixture
    receipt = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert receipt.installed
    installation = getattr(f.app, install._INSTALLED_ATTR)
    assert installation.adapter.profile.max_turn_seconds == 720.0


@pytest.mark.parametrize("raw", ["", "abc", "0", "-30", "10", "99999", "nan", "1e999"])
async def test_max_turn_window_env_invalid_declines_install_fail_closed(
        transaction_fixture, monkeypatch, raw):
    """A declared-but-invalid window never silently falls back to the default:
    the native install declines with the fixed refusal code and nothing mounts."""
    monkeypatch.setenv("OCC3_MAX_TURN_SECONDS", raw)
    f = transaction_fixture
    receipt = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert not receipt.installed and receipt.code == "max_turn_seconds_invalid"
    assert getattr(f.app, install._INSTALLED_ATTR, None) is None


async def test_max_turn_window_env_never_overrides_explicit_profile(
        transaction_fixture, monkeypatch):
    """An explicit caller-supplied buffer profile wins untouched: the boot
    override only replaces the default profile, it must not mutate a profile
    the trusted caller constructed deliberately."""
    monkeypatch.setenv("OCC3_MAX_TURN_SECONDS", "720")
    from occ3_sglang.chat_adapter import BufferProfile
    explicit = BufferProfile(max_turn_seconds=300)
    f = transaction_fixture
    receipt = await install.install_native_runtime(f.app, f.tm,
        experimental_opt_in=True, buffer_profile=explicit)
    assert receipt.installed
    installation = getattr(f.app, install._INSTALLED_ATTR)
    assert installation.adapter.profile.max_turn_seconds == 300
    assert installation.adapter.profile is explicit
