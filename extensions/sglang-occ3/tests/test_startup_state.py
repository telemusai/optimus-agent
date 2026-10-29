"""Phase-split control plane: truthful state machine, no GPU/socket use.

The app, auth middleware and install transaction are the offline fixtures
from test_native_install; the native backend replies are explicit synthetic
metadata only. Passing is never live GPU or capability proof.
"""
from __future__ import annotations

import asyncio
import logging
import re
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from starlette.requests import Request

from occ3_sglang import native_install as install
from occ3_sglang.native_profile import advertised_methods
from test_native_install import KEY, pinned_auth, transaction_fixture

AUTH = {"Authorization": "Bearer " + KEY}
IDENTITY_REQUEST = {"schema": "occ3.runtime.identity.request.v1"}
IDENTITY_FIELDS = {"schema", "runtime_id", "runtime_generation", "model_fingerprint",
    "method_id", "tp_layout_digest", "layer_index", "hidden_dim", "dtype",
    "expected_ranks", "ready_ranks", "profile_digest", "source_capture",
    "continuous_prefix", "terminal_journal", "server_time_epoch", "max_ttl_seconds",
    "max_capture_bytes"}
STATUS_KEYS = {"schema", "state", "code", "stage", "attempt", "experimental_selected",
    "preflight_reached", "backend_constructed", "worker_readiness_reached",
    "hooks_prepared", "route_mount_attempted", "rollback_completed", "plugin",
    "version", "enabled", "model_fingerprint", "channels", "metrics"}
CAPABILITY_KEYS = {"schema", "supported", "reason", "experimental_prompt_kv_path",
    "model_fingerprint", "connector_id", "methods", "experimental_baselines",
    "protocol", "api_version", "same_model_only", "verification", "cache", "native"}
PROGRESS_BOOLEANS = ("experimental_selected", "preflight_reached", "backend_constructed",
    "worker_readiness_reached", "hooks_prepared", "route_mount_attempted")


@pytest.fixture(autouse=True)
def machine():
    from occ3_sglang import startup_state

    startup_state.reset()
    yield startup_state
    startup_state.reset()


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


def path_counts(app):
    counts = {}
    for route in app_routes(app):
        path = getattr(route, "path", None)
        counts[path] = counts.get(path, 0) + 1
    return counts


@pytest.fixture
def phase(stub_http_server, transaction_fixture, monkeypatch):
    """Registered control plane on the pinned-auth transaction app."""
    from occ3_sglang import startup_state

    f = transaction_fixture
    module_file = Path(startup_state.__file__).resolve()
    monkeypatch.setattr(install, "_verify_control_plane_sources",
                        lambda: {str(module_file): module_file.read_text(encoding="utf-8")})
    stub_http_server._global_state = NS(tokenizer_manager=f.tm)
    stub_http_server.app = f.app
    f.tm.served_model_name = "fixture-model"
    assert install.register_control_plane(f.app, f.tm) is True
    assert startup_state.current().state.value == "CONTROL_REGISTERED"
    return f


@pytest.fixture
def serve_flow(pinned_auth, monkeypatch, machine):
    """Selected native startup on a stub server with real registration."""
    app = FastAPI()
    tm = NS(server_args=NS(api_key=KEY))
    pinned_auth.add_api_key_middleware(app, api_key=KEY, admin_api_key=None)
    events = []

    class Server:
        async def serve(self, *args, **kwargs):
            events.append("serve")
            return "native-result"

    server = Server()
    monkeypatch.setattr(install, "_read_startup_literal", lambda: {"EXPERIMENTAL_STARTUP": True})
    monkeypatch.setattr(install, "_verify_startup_artifacts", lambda selection: None)
    monkeypatch.setattr(install, "_verify_native_server", lambda s, a, t: s.serve)
    module_file = Path(machine.__file__).resolve()
    monkeypatch.setattr(install, "_verify_control_plane_sources",
                        lambda: {str(module_file): module_file.read_text(encoding="utf-8")})
    return NS(app=app, tm=tm, server=server, events=events)


async def test_registration_before_install_answers_not_ready_then_full_tuple(phase):
    from test_latent_backend import health as backend_health

    f = phase
    counts = path_counts(f.app)
    for path in ("/occ3/v1/status", "/occ3/v1/capabilities",
                 "/occ3/v1/native/runtime/identity"):
        assert counts.get(path) == 1
    async with AsyncClient(transport=ASGITransport(app=f.app), base_url="http://fixture") as client:
        pre = await client.post("/occ3/v1/native/runtime/identity",
                                json=IDENTITY_REQUEST, headers=AUTH)
        assert pre.status_code == 503 and pre.json() == {"error": {"code": "not_ready"}}
        assert pre.headers["cache-control"] == "no-store"
        assert "runtime_epoch" not in pre.text
        status = await client.get("/occ3/v1/status", headers=AUTH)
        assert status.status_code == 200 and status.json()["state"] == "CONTROL_REGISTERED"
    # Real clients cannot reach the app before uvicorn serves; this probe is a
    # test-only interleaving, so restore the pre-serving stack install requires.
    f.app.middleware_stack = None
    receipt = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert receipt.installed
    backend = f.created[0].backend
    backend.profile = f.profile

    async def fixture_refresh():
        return backend_health(f.profile)

    backend.refresh_readiness = fixture_refresh
    async with AsyncClient(transport=ASGITransport(app=f.app), base_url="http://fixture") as client:
        ident = await client.post("/occ3/v1/native/runtime/identity",
                                  json=IDENTITY_REQUEST, headers=AUTH)
        assert ident.status_code == 200, ident.text
        body = ident.json()
        assert set(body) == IDENTITY_FIELDS and body["schema"] == "occ3.runtime.identity.v1"
        assert body["runtime_id"] == f.profile.runtime_epoch
        assert body["runtime_generation"] == f.profile.runtime_generation
        assert body["model_fingerprint"] == f.profile.model_fingerprint
        assert body["expected_ranks"] == body["ready_ranks"] == 8
        assert body["source_capture"] and body["continuous_prefix"] and body["terminal_journal"]
        assert body["max_capture_bytes"] == 128 * 1024 * 1024  # server wire cap advertised
        assert ident.headers["cache-control"] == "no-store"
        status = await client.get("/occ3/v1/status", headers=AUTH)
        payload = status.json()
        assert payload["state"] == "NATIVE_READY"
        assert payload["code"] == "installed_experimental_not_live_validated"
        assert payload["stage"] == "install" and re.fullmatch("[0-9a-f]{8}", payload["attempt"])
        assert all(payload[name] for name in PROGRESS_BOOLEANS)
        assert payload["rollback_completed"] is False
        assert payload["enabled"] is False


async def test_declined_status_is_truthful(phase, monkeypatch):
    f = phase

    async def refuse(app, tm, key, buffer_profile):
        raise install.InstallationError("source_digest_mismatch")

    monkeypatch.setattr(install, "_construct_components", refuse)
    receipt = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert not receipt.installed and receipt.code == "source_digest_mismatch"
    async with AsyncClient(transport=ASGITransport(app=f.app), base_url="http://fixture") as client:
        status = await client.get("/occ3/v1/status", headers=AUTH)
        payload = status.json()
        assert set(payload) == STATUS_KEYS
        assert payload["state"] == "DECLINED" and payload["code"] == "source_digest_mismatch"
        assert payload["stage"] == "install"
        assert payload["experimental_selected"] is True
        assert payload["preflight_reached"] is True
        assert payload["backend_constructed"] is False
        assert payload["rollback_completed"] is True  # nothing was mutated
        caps = await client.get("/occ3/v1/capabilities", headers=AUTH)
        advertisement = caps.json()
        assert advertisement["supported"] is False
        assert advertisement["methods"] == list(advertised_methods())
        assert advertisement["reason"] == "native_install_declined"
        assert advertisement["native"]["installed"] is False
        assert advertisement["native"]["state"] == "DECLINED"
        ident = await client.post("/occ3/v1/native/runtime/identity",
                                  json=IDENTITY_REQUEST, headers=AUTH)
        assert ident.status_code == 503 and ident.json() == {"error": {"code": "not_ready"}}
        ordinary = await client.post("/v1/chat/completions", headers=AUTH,
                                     json={"messages": [{"role": "user", "content": "still serving"}],
                                           "max_tokens": 16})
        assert ordinary.status_code == 200
        assert ordinary.json() == {"ordinary": [{"role": "user", "content": "still serving"}],
                                   "max_new_tokens": 16}


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"},
                                      {"Authorization": "Basic " + KEY}])
async def test_control_routes_require_the_original_key(phase, headers):
    f = phase
    async with AsyncClient(transport=ASGITransport(app=f.app), base_url="http://fixture") as client:
        assert (await client.get("/occ3/v1/status", headers=headers)).status_code == 401
        assert (await client.get("/occ3/v1/capabilities", headers=headers)).status_code == 401
        denied = await client.post("/occ3/v1/native/runtime/identity",
                                  json=IDENTITY_REQUEST, headers=headers)
        assert denied.status_code == 401


async def test_valid_key_gets_status_and_capabilities_but_never_identity(phase):
    f = phase
    async with AsyncClient(transport=ASGITransport(app=f.app), base_url="http://fixture") as client:
        assert (await client.get("/occ3/v1/status", headers=AUTH)).status_code == 200
        assert (await client.get("/occ3/v1/capabilities", headers=AUTH)).status_code == 200
        denied = await client.post("/occ3/v1/native/runtime/identity",
                                   json=IDENTITY_REQUEST, headers=AUTH)
        assert denied.status_code == 503 and denied.json() == {"error": {"code": "not_ready"}}


async def test_post_install_identity_without_broker_marking_is_refused(phase):
    from test_latent_backend import health as backend_health

    f = phase
    f.app.middleware_stack = None  # no HTTP probe happened; keep the pre-serving stack
    receipt = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert receipt.installed
    backend = f.created[0].backend
    backend.profile = f.profile

    async def fixture_refresh():
        return backend_health(f.profile)

    backend.refresh_readiness = fixture_refresh
    route = next(route for route in app_routes(f.app)
                 if route.path == "/occ3/v1/native/runtime/identity")

    async def receive():
        return {"type": "http.request", "body": b'{"schema": "occ3.runtime.identity.request.v1"}',
                "more_body": False}

    request = Request({"type": "http", "method": "POST",
                       "path": "/occ3/v1/native/runtime/identity",
                       "headers": [], "state": {}}, receive=receive)
    response = await route.endpoint(request)
    assert response.status_code == 401
    assert b'"identity"' in response.body  # fixed authority code, no tuple


@pytest.mark.parametrize("family,setup,code", [
    ("preflight", lambda f: setattr(f.tm.request_logger, "log_requests", True),
     "unsupported_frontend_profile"),
    ("attestation", "construct", "source_digest_mismatch"),
    ("hooks", lambda f: f.hooks.HookRegistry._patched.add(install._EXPECTED_HOOKS[0][0]),
     "hook_target_occupied"),
    ("readiness", lambda f: setattr(f.controls, "ready", False), "backend_not_ready"),
    ("control", lambda f: setattr(f.tm, "_occ3_control_client", object()),
     "control_client_occupied"),
    ("cleanup", "mount-then-unclosed", "installation_cleanup_unconfirmed"),
])
async def test_no_capability_advertisement_on_decline(phase, monkeypatch, family, setup, code):
    f = phase
    if setup == "construct":
        async def refuse(app, tm, key, buffer_profile):
            raise install.InstallationError(code)

        monkeypatch.setattr(install, "_construct_components", refuse)
    elif setup == "mount-then-unclosed":
        original = f.app.add_middleware

        def broken(*args, **kwargs):
            original(*args, **kwargs)
            raise RuntimeError(KEY)

        monkeypatch.setattr(f.app, "add_middleware", broken)

        async def unfinished(components):
            return False

        monkeypatch.setattr(install, "_close_components", unfinished)
    else:
        setup(f)
    result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert not result.installed and result.code == code
    async with AsyncClient(transport=ASGITransport(app=f.app), base_url="http://fixture") as client:
        status = await client.get("/occ3/v1/status", headers=AUTH)
        assert status.json()["state"] == "DECLINED" and status.json()["code"] == code
        caps = await client.get("/occ3/v1/capabilities", headers=AUTH)
        advertisement = caps.json()
        assert advertisement["supported"] is False
        assert advertisement["methods"] == list(advertised_methods())
        assert advertisement["reason"] == "native_install_declined"
        assert advertisement["native"]["installed"] is False
        assert advertisement["native"]["source_ready"] is False
        assert advertisement["native"]["receiver_ready"] is False
        assert advertisement["native"]["live_validated"] is False
        ident = await client.post("/occ3/v1/native/runtime/identity",
                                  json=IDENTITY_REQUEST, headers=AUTH)
        assert ident.status_code == 503 and "runtime_epoch" not in ident.text


async def test_state_machine_lock_holder_and_terminal_transitions(phase, machine):
    f = phase
    # A non-holder attempt never mutates the machine.
    install._install_lock.acquire()
    try:
        busy = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
        assert busy.code == "installation_busy" and not busy.installed
        assert machine.current().state.value == "CONTROL_REGISTERED"
        assert machine.current().code == "installation_busy"
    finally:
        install._install_lock.release()
    # Opt-in refusal records its fixed code without transitioning.
    refused = await install.install_native_runtime(f.app, f.tm)
    assert refused.code == "explicit_experimental_opt_in_required"
    assert machine.current().state.value == "CONTROL_REGISTERED"
    assert machine.current().code == "explicit_experimental_opt_in_required"


def test_not_selected_prepare_stays_disabled_with_code(machine, monkeypatch):
    class Server:
        async def serve(self):
            return "native-result"

    app = FastAPI()
    tm = NS(server_args=NS(api_key=KEY))
    monkeypatch.setattr(install, "_read_startup_literal", lambda: None)
    receipt = install.prepare_selected_native_server(Server(), app, tm)
    assert not receipt.selected and receipt.code == "experimental_startup_not_selected"
    current = machine.current()
    assert current.state.value == "DISABLED"
    assert current.code == "experimental_startup_not_selected"
    assert current.stage == "prepare" and current.experimental_selected is False


async def test_cancelled_installer_supersedes_to_native_startup_aborted(serve_flow, machine, monkeypatch):
    f = serve_flow

    async def cancelled(app, tm, **kwargs):
        raise asyncio.CancelledError

    monkeypatch.setattr(install, "install_native_runtime", cancelled)
    assert install.prepare_selected_native_server(f.server, f.app, f.tm).prepared
    with pytest.raises(asyncio.CancelledError):
        await f.server.serve()
    current = machine.current()
    assert current.state.value == "DECLINED" and current.code == "native_startup_aborted"
    assert current.stage == "install" and current.attempt is not None


async def test_uncertain_refusal_supersedes_recorded_code(serve_flow, machine, monkeypatch):
    f = serve_flow

    async def refuse(app, tm, **kwargs):
        return install.InstallationReceipt(False, "installation_busy")

    monkeypatch.setattr(install, "install_native_runtime", refuse)
    assert install.prepare_selected_native_server(f.server, f.app, f.tm).prepared
    with pytest.raises(install.InstallationError, match="native_startup_cleanup_unconfirmed"):
        await f.server.serve()
    current = machine.current()
    assert current.state.value == "DECLINED"
    assert current.code == "native_startup_cleanup_unconfirmed"


async def test_verified_safe_refusal_records_declined_code(phase, machine, caplog):
    f = phase
    f.controls.ready = False
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert not result.installed and result.code == "backend_not_ready"
    current = machine.current()
    assert current.state.value == "DECLINED" and current.code == "backend_not_ready"
    assert current.experimental_selected is True and current.preflight_reached is True
    assert current.worker_readiness_reached is True and current.route_mount_attempted is False
    # The state record carries the attempt marker of the matching C30 line.
    line = next(record.getMessage() for record in caplog.records
                if "native installation decline" in record.getMessage())
    assert current.attempt == re.search(r"attempt=([0-9a-f]{8})", line).group(1)


async def test_payload_hygiene_fixed_keys_and_no_secret_leakage(phase, monkeypatch, caplog):
    f = phase
    secret = "SYNTHETIC-STARTUP-SECRET-7301"

    async def refuse(app, tm, key, buffer_profile):
        raise RuntimeError(secret)

    monkeypatch.setattr(install, "_construct_components", refuse)
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        result = await install.install_native_runtime(f.app, f.tm, experimental_opt_in=True)
    assert result.code == "installation_unavailable" and not result.installed
    async with AsyncClient(transport=ASGITransport(app=f.app), base_url="http://fixture") as client:
        status = (await client.get("/occ3/v1/status", headers=AUTH)).json()
        caps = (await client.get("/occ3/v1/capabilities", headers=AUTH)).json()
    assert set(status) == STATUS_KEYS and set(caps) == CAPABILITY_KEYS
    assert secret not in str(status) + str(caps) + caplog.text
    assert status["code"] == "installation_unavailable"
    assert caps["native"]["code"] == "installation_unavailable"


def test_dual_registration_mounts_each_route_exactly_once(stub_http_server, stub_tm,
                                                           pinned_auth, monkeypatch):
    from occ3_sglang import plugin, startup_state

    app = FastAPI()
    stub_tm.server_args.api_key = KEY
    pinned_auth.add_api_key_middleware(app, api_key=KEY, admin_api_key=None)
    stub_http_server.app = app
    stub_http_server._global_state = NS(tokenizer_manager=stub_tm)
    plugin._LOADED = False
    plugin.load()
    assert plugin._LOADED and getattr(app, startup_state.CONTROL_PLANE_ATTR) is True
    module_file = Path(startup_state.__file__).resolve()
    monkeypatch.setattr(install, "_verify_control_plane_sources",
                        lambda: {str(module_file): module_file.read_text(encoding="utf-8")})
    assert install.register_control_plane(app, stub_tm) is True
    counts = path_counts(app)
    for path in startup_state.CONTROL_PLANE_PATHS:
        assert counts.get(path) == 1, path
    assert startup_state.current().state.value == "CONTROL_REGISTERED"


def test_registration_refused_without_verified_original_auth(machine, caplog):
    app = FastAPI()  # no original api-key middleware at all
    tm = NS(server_args=NS(api_key=KEY))
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        assert install.register_control_plane(app, tm) is False
    current = machine.current()
    assert current.state.value == "DISABLED"
    assert current.code == "original_auth_registration_required"
    assert current.stage == "prepare"
    records = [record for record in caplog.records
               if record.name == "occ3_sglang.native_install"]
    assert len(records) == 1 and records[0].levelno == logging.ERROR
    assert re.fullmatch(r"occ3-bridge: control plane registration decline "
                        r"\(attempt=[0-9a-f]{8} stage=prepare code=original_auth_registration_required\)",
                        records[0].getMessage())
    assert not getattr(app, "_occ3_control_plane_installed", False)
    assert all(getattr(route, "path", None) != "/occ3/v1/status"
               for route in app.router.routes)


def test_registration_refused_on_attestation_mismatch(machine, monkeypatch, tmp_path, caplog,
                                                       pinned_auth):
    app = FastAPI()
    tm = NS(server_args=NS(api_key=KEY))
    pinned_auth.add_api_key_middleware(app, api_key=KEY, admin_api_key=None)
    def incomplete():
        raise install.InstallationError("source_attestation_incomplete")

    monkeypatch.setattr(install, "_verify_control_plane_sources", incomplete)
    with caplog.at_level(logging.ERROR, logger="occ3_sglang.native_install"):
        assert install.register_control_plane(app, tm) is False
    current = machine.current()
    assert current.state.value == "DISABLED"
    assert current.code == "source_attestation_incomplete"
    assert not getattr(app, "_occ3_control_plane_installed", False)
    # No control route was mounted on the refused app.
    assert all(getattr(route, "path", None) != "/occ3/v1/status"
               for route in app.router.routes)


def test_registration_refusal_restores_route_list(machine, monkeypatch, pinned_auth):
    app = FastAPI()
    tm = NS(server_args=NS(api_key=KEY))
    pinned_auth.add_api_key_middleware(app, api_key=KEY, admin_api_key=None)
    module_file = Path(machine.__file__).resolve()
    monkeypatch.setattr(install, "_verify_control_plane_sources",
                        lambda: {str(module_file): module_file.read_text(encoding="utf-8")})
    before = list(app.router.routes)
    original = app.include_router

    def broken(router, *args, **kwargs):
        raise RuntimeError("SYNTHETIC-MOUNT-SECRET")

    monkeypatch.setattr(app, "include_router", broken)
    assert install.register_control_plane(app, tm) is False
    assert app.router.routes == before
    assert machine.current().code == "experimental_startup_refused"
    monkeypatch.setattr(app, "include_router", original)
    assert install.register_control_plane(app, tm) is True
    assert path_counts(app).get("/occ3/v1/status") == 1
    assert machine.current().state.value == "CONTROL_REGISTERED"
