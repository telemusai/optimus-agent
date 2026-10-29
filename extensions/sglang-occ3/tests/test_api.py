"""Route behavior with a stubbed tokenizer manager and http_server."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from occ3_sglang import api, plugin
from occ3_sglang.registry import registry


@pytest.fixture
def client(stub_http_server, stub_tm):
    from occ3_sglang import startup_state

    app = FastAPI()
    app.include_router(api.router)
    # Dormant parity surface: the control plane owns status/capabilities now.
    app.include_router(startup_state.create_control_router())
    stub_http_server.app = app
    stub_http_server._global_state = type("S", (), {"tokenizer_manager": stub_tm})()
    with TestClient(app) as c:
        yield c


@pytest.fixture(autouse=True)
def clean_startup_state():
    from occ3_sglang import startup_state

    startup_state.reset()
    yield
    startup_state.reset()


@pytest.fixture(autouse=True)
def clean_registry():
    # The registry is a module singleton; reset between tests.
    with registry()._lock:
        registry()._channels.clear()
    yield
    with registry()._lock:
        registry()._channels.clear()


def test_status_reports_plugin_and_identity(client):
    r = client.get("/occ3/v1/status")
    assert r.status_code == 200
    body = r.json()
    assert body["plugin"] == "occ3-bridge"
    assert body["enabled"] is False
    assert body["model_fingerprint"]
    assert "channels" in body and "metrics" in body


def test_status_payload_is_a_fixed_key_superset_with_dormant_state(client):
    r = client.get("/occ3/v1/status")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "schema", "state", "code", "stage", "attempt",
        "experimental_selected", "preflight_reached", "backend_constructed",
        "worker_readiness_reached", "hooks_prepared", "route_mount_attempted",
        "rollback_completed", "plugin", "version", "enabled",
        "model_fingerprint", "channels", "metrics"}
    assert body["schema"] == "optimus.occ3.status.v1"
    assert body["state"] == "DISABLED"  # dormant mount never transitions
    assert body["code"] is None and body["stage"] is None and body["attempt"] is None
    import occ3_sglang
    assert body["version"] == occ3_sglang.__version__
    assert not any(body[name] for name in (
        "experimental_selected", "preflight_reached", "backend_constructed",
        "worker_readiness_reached", "hooks_prepared", "route_mount_attempted",
        "rollback_completed"))


def test_identity_route_exposes_contract_fields(client):
    r = client.get("/occ3/v1/identity")
    assert r.status_code == 200
    body = r.json()
    assert body["fields"]["architecture"] == "GlmMoeDsaForCausalLM"
    assert len(body["field_order"]) == 11
    assert body["fingerprint"].count("|") == 10


def test_capabilities_advertisement_shape(client):
    r = client.get("/occ3/v1/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert body["supported"] is False
    assert body["reason"] == "native_latent_receiver_not_implemented"
    assert body["connector_id"] == "sglang-occ3"
    from occ3_sglang.native_profile import advertised_methods
    assert body["methods"] == list(advertised_methods())
    assert body["same_model_only"] is True
    assert body["verification"] == "cached_tokens"
    assert body["model_fingerprint"]


def test_capabilities_payload_is_a_fixed_key_superset_with_native_block(client):
    r = client.get("/occ3/v1/capabilities")
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "schema", "supported", "reason", "experimental_prompt_kv_path",
        "model_fingerprint", "connector_id", "methods", "experimental_baselines",
        "protocol", "api_version", "same_model_only", "verification", "cache",
        "native"}
    assert set(body["native"]) == {"state", "code", "installed", "source_ready",
                                   "receiver_ready", "live_validated"}
    assert body["native"] == {"state": "DISABLED", "code": None, "installed": False,
                              "source_ready": False, "receiver_ready": False,
                              "live_validated": False}
    from occ3_sglang.native_profile import advertised_methods
    assert body["methods"] == list(advertised_methods()) and body["supported"] is False


def test_dormant_identity_answers_not_ready_without_state(client):
    r = client.post("/occ3/v1/native/runtime/identity",
                    json={"schema": "occ3.runtime.identity.request.v1"})
    assert r.status_code == 503
    assert r.json() == {"error": {"code": "not_ready"}}
    assert r.headers["cache-control"] == "no-store"
    assert "runtime_epoch" not in r.text


@pytest.mark.parametrize("path,payload", [
    ("send", {"text": "hello latent world"}),
    ("send", {"input_ids": [1, 2], "embeds": [[1.0]], "positions": [0]}),
    ("send", {"opt_in": True, "session_id": "claimed", "text": "still legacy"}),
    ("send", {"invalid": "synthetic-secret"}),
    ("recv", {"channel": "old", "suffix": "answer briefly"}),
    ("release", {"channel": "old"}),
])
def test_legacy_mutations_are_quarantined_without_inference(client, stub_tm, path, payload, caplog):
    result = client.post("/occ3/v1/" + path, json=payload)
    assert result.status_code == 410
    assert result.json()["detail"] == "legacy_unbound_api_disabled_use_opt_in_handles"
    assert stub_tm.requests == []
    assert "synthetic-secret" not in result.text + caplog.text


def test_legacy_channels_do_not_enumerate_unbound_handles(client):
    registry().create("unbound", [1, 2, 3], 60)
    response = client.get("/occ3/v1/channels")
    assert response.json()["channels"] == []
    assert response.json()["available"] is False


def test_plugin_load_installs_router_once(stub_http_server, stub_tm):
    app = FastAPI()
    stub_http_server.app = app
    stub_http_server._global_state = type("S", (), {"tokenizer_manager": stub_tm})()
    plugin._LOADED = False
    plugin.load()
    plugin.load()  # idempotent
    assert getattr(app, "_occ3_router_installed") is True
    with TestClient(app) as c:
        r = c.get("/occ3/v1/status")
        assert r.status_code == 200
        assert r.json()["plugin"] == "occ3-bridge"
        r = c.get("/occ3/v1/capabilities")
        assert r.status_code == 200


# These fixtures simulate cache tiers. Passing them is not live GPU proof.
def kv_bound(client, session="sender", **extra):
    fp = client.get("/occ3/v1/latent/capabilities").json()["model_fingerprint"]
    return {"opt_in": True, "session_id": session, "model_fingerprint": fp, **extra}


def kv_capture_and_send(client, text='Facts {"code":"alpha"}', ttl=60.0):
    cap = client.post("/occ3/v1/latent/capture", json=kv_bound(client, text=text, ttl_s=ttl))
    assert cap.status_code == 200, cap.text
    source = cap.json()
    sent = client.post("/occ3/v1/latent/send", json=kv_bound(
        client, capture_handle=source["capture_handle"], receiver_session_id="receiver"))
    assert sent.status_code == 200, sent.text
    return source, sent.json()


def kv_receive(client, sent, **extra):
    body = kv_bound(client, session="receiver", channel_handle=sent["channel_handle"],
                    suffix="Return source facts", max_new_tokens=64)
    body.update(extra)
    return client.post("/occ3/v1/latent/recv", json=body)


def test_handle_source_capture_then_zero_inference_send_then_receive(kv_client):
    client, tm, clock = kv_client
    source, sent = kv_capture_and_send(client)
    assert len(tm.kv_requests) == 1  # binding does not prefill again
    assert "carrier_token_ids" not in source and "salt" not in source
    source_req = tm.kv_requests[0]
    assert source_req.rid == source["source_request_id"]
    assert len(source_req.input_ids) == source["source_prompt_tokens"]
    assert source["guard_tokens"] == 2 * tm.server_args.page_size
    got = kv_receive(client, sent)
    assert got.status_code == 200, got.text
    result = got.json()
    assert result["hit"] and result["fallback_used"] is False
    assert result["cache_tiers"]["device"] >= source["required_gpu_tokens"]
    assert result["cache_tiers"]["device"] < source["source_prompt_tokens"]  # page/bigram guard tail
    assert tm.kv_requests[1].cache_salt == source_req.cache_salt
    assert tm.kv_requests[1].input_ids[:len(source_req.input_ids)] == source_req.input_ids
    assert tm.kv_requests[1].session_id == "receiver"
    assert tm.kv_requests[1].routed_dp_rank == 0
    assert kv_receive(client, sent).status_code == 403  # one-shot, no duplicate inference
    assert len(tm.kv_requests) == 2


@pytest.mark.parametrize("mode", ["missing_tier", "host", "storage", "retracted", "missing_retractions", "miss", "abort"])
def test_handle_cache_recomputation_never_becomes_success(kv_client, mode):
    client, tm, clock = kv_client
    source, sent = kv_capture_and_send(client)
    def mutate(obj, out):
        meta = out["meta_info"]
        if mode == "missing_tier":
            meta.pop("cached_tokens_details")
        elif mode == "host":
            meta["cached_tokens_details"]["host"] = 1
        elif mode == "storage":
            meta["cached_tokens_details"]["storage"] = 1
        elif mode == "retracted":
            meta["num_retractions"] = 1
        elif mode == "missing_retractions":
            meta.pop("num_retractions")
        elif mode == "miss":
            meta["cached_tokens"] = 0
            meta["cached_tokens_details"] = {"device": 0, "host": 0}
        else:
            meta["finish_reason"] = {"type": "abort"}
    tm.mutate_output = mutate
    out = kv_receive(client, sent)
    assert out.status_code == 409
    assert "output_text" not in out.json()
    assert "alpha" not in out.text
    tm.mutate_output = None
    assert kv_receive(client, sent).status_code == 410  # cannot warm then retry
    assert len(tm.kv_requests) == 2


def test_handle_session_and_exact_model_bindings(kv_client):
    client, tm, clock = kv_client
    source, sent = kv_capture_and_send(client)
    assert kv_receive(client, sent, session_id="unrelated").status_code == 403
    assert kv_receive(client, sent, model_fingerprint=source["model_fingerprint"] + "-different").status_code == 409
    assert len(tm.kv_requests) == 1
    assert kv_receive(client, sent).status_code == 200


def test_handle_same_text_uses_distinct_source_salts(kv_client):
    client, tm, clock = kv_client
    first, first_sent = kv_capture_and_send(client)
    second, second_sent = kv_capture_and_send(client)
    assert first["capture_handle"] != second["capture_handle"]
    assert first_sent["channel_handle"] != second_sent["channel_handle"]
    assert first["salt_sha256"] != second["salt_sha256"]
    assert first["source_cached_tokens"] == second["source_cached_tokens"] == 0
    assert tm.kv_requests[0].input_ids == tm.kv_requests[1].input_ids
    assert tm.kv_requests[0].cache_salt != tm.kv_requests[1].cache_salt


def test_handle_expiry_release_and_generation_staleness(kv_client, monkeypatch):
    from occ3_sglang import kv_handles
    client, tm, clock = kv_client
    source, sent = kv_capture_and_send(client, ttl=1.0)
    clock[0] += 1.01
    assert kv_receive(client, sent).status_code == 410
    assert not kv_handles.store.captures
    source, sent = kv_capture_and_send(client)
    released = client.post("/occ3/v1/latent/release", json=kv_bound(client, capture_handle=source["capture_handle"]))
    assert released.status_code == 200 and released.json()["gpu_erased"] is False
    assert released.json()["cpu_references_removed"] is True
    assert kv_receive(client, sent).status_code == 410
    source, sent = kv_capture_and_send(client)
    old_epoch = source["process_epoch"]
    monkeypatch.setattr(kv_handles, "store", kv_handles.CaptureStore(clock=lambda: clock[0]))
    assert kv_handles.store.epoch != old_epoch
    assert kv_receive(client, sent).status_code == 410
    assert len(tm.kv_requests) == 3  # no restart or model operations in this test


def test_handle_send_rejects_plaintext_and_requires_opt_in(kv_client):
    client, tm, clock = kv_client
    source, sent = kv_capture_and_send(client)
    payload = kv_bound(client, capture_handle=source["capture_handle"], receiver_session_id="other", text="fallback")
    assert client.post("/occ3/v1/latent/send", json=payload).status_code == 422
    assert client.post("/occ3/v1/latent/capture", json=kv_bound(client, opt_in=False, text="no consent")).status_code == 403
    assert len(tm.kv_requests) == 1


def test_handle_runtime_and_per_session_quota_fail_closed(kv_client):
    client, tm, clock = kv_client
    for _ in range(4):
        kv_capture_and_send(client)
    extra = client.post("/occ3/v1/latent/capture", json=kv_bound(client, text="excess"))
    assert extra.status_code == 429
    assert len(tm.kv_requests) == 4
    tm.server_args.device = "cpu"
    caps = client.get("/occ3/v1/latent/capabilities").json()
    assert caps["available"] is False and caps["native_receiver_integrated"] is False
    assert caps["true_latent_method"] is False
    assert client.post("/occ3/v1/latent/capture", json=kv_bound(client, session="other", text="excess")).status_code == 503


def test_handle_source_and_receiver_errors_do_not_leak_private_body(kv_client):
    client, tm, clock = kv_client
    source, sent = kv_capture_and_send(client)
    tm.fail_generation = True
    out = kv_receive(client, sent)
    assert out.status_code == 502 and "synthetic-private" not in out.text
    assert kv_receive(client, sent).status_code == 410
    out = client.post("/occ3/v1/latent/capture", json=kv_bound(client, text="new source"))
    assert out.status_code == 502 and "synthetic-private" not in out.text


@pytest.mark.parametrize("field,value", [("id", "unrelated-request"),
                                          ("weight_version", "other-weights"),
                                          ("weight_versions", ["mid-request-update"])])
def test_handle_receiver_rejects_wrong_actual_request_or_weight_epoch(kv_client, field, value):
    client, tm, clock = kv_client
    source, sent = kv_capture_and_send(client)
    tm.mutate_output = lambda obj, out: out["meta_info"].update({field: value})
    out = kv_receive(client, sent)
    assert out.status_code == 409
    assert "output_text" not in out.json()


def test_handle_receiver_rejects_impossible_device_counter(kv_client):
    client, tm, clock = kv_client
    source, sent = kv_capture_and_send(client)
    def mutate(obj, out):
        fake = out["meta_info"]["prompt_tokens"] + 100
        out["meta_info"]["cached_tokens"] = fake
        out["meta_info"]["cached_tokens_details"]["device"] = fake
    tm.mutate_output = mutate
    assert kv_receive(client, sent).status_code == 409


@pytest.mark.parametrize("operation", ["capture", "recv"])
@pytest.mark.parametrize("error_type", ["http", "ordinary"])
def test_backend_errors_and_logs_never_expose_secret(kv_client, caplog, operation, error_type):
    from fastapi import HTTPException
    from occ3_sglang import kv_handles
    client, tm, clock = kv_client
    secret = "SYNTHETIC-BACKEND-SECRET-9845"
    if operation == "recv":
        source, sent = kv_capture_and_send(client)
    async def broken(obj, request=None):
        if error_type == "http":
            raise HTTPException(418, secret, headers={"secret": secret})
        raise RuntimeError(secret)
        yield  # This is a deliberately failing async generator.
    tm.generate_request = broken
    if operation == "capture":
        result = client.post("/occ3/v1/latent/capture", json=kv_bound(client, text="source"))
    else:
        result = kv_receive(client, sent)
    assert result.status_code == 502
    assert result.json()["detail"] == "backend_failed_cleanup_unconfirmed"
    assert secret not in result.text + caplog.text + str(result.headers)
    assert len(tm.abort_requests) == 1
    assert tm.abort_requests[0].rid.startswith("occ3-")
    assert tm.abort_requests[0].abort_all is False
    pending = list(kv_handles.store.captures.values())
    assert len(pending) == 1 and pending[0].state == "UNCERTAIN"
    assert pending[0].abort_admitted and pending[0].local_task_closed
    assert pending[0].input_ids == []


def test_tokenizer_http_exception_is_fixed_and_private(kv_client, caplog):
    from fastapi import HTTPException
    client, tm, clock = kv_client
    def broken(*args, **kwargs):
        raise HTTPException(400, "SYNTHETIC-TOKENIZER-SECRET")
    tm.tokenizer.encode = broken
    result = client.post("/occ3/v1/latent/capture", json=kv_bound(client, text="source"))
    assert result.status_code == 502 and result.json()["detail"] == "tokenizer_failed"
    assert "SYNTHETIC-TOKENIZER-SECRET" not in result.text + caplog.text
    assert tm.kv_requests == []


def test_plugin_failed_install_can_retry_without_duplicate_routes(stub_http_server, stub_tm, monkeypatch, caplog):
    from occ3_sglang import capture
    app = FastAPI()
    original = app.include_router
    calls = [0]
    def fail_second(router, *args, **kwargs):
        calls[0] += 1
        if calls[0] == 2:
            raise RuntimeError("SYNTHETIC-INSTALL-SECRET")
        return original(router, *args, **kwargs)
    stub_http_server.app = app
    plugin._LOADED = False
    before = list(app.router.routes)
    monkeypatch.setattr(app, "include_router", fail_second)
    plugin.load()
    assert not plugin._LOADED and app.router.routes == before
    assert not getattr(app, "_occ3_router_installed", False)
    monkeypatch.setattr(app, "include_router", original)
    monkeypatch.setattr(capture, "register_capture_hook", lambda: None)
    plugin.load()
    assert plugin._LOADED
    from occ3_sglang import kv_handles
    expected = FastAPI()
    from occ3_sglang import startup_state
    expected.include_router(api.router)
    expected.include_router(kv_handles.router)
    expected.include_router(startup_state.create_control_router())
    assert len(app.router.routes) == len(expected.router.routes)
    plugin.load()
    assert len(app.router.routes) == len(expected.router.routes)
    assert "/occ3/v1/status" in app.openapi()["paths"]
    assert "/occ3/v1/latent/capture" in app.openapi()["paths"]
    assert "SYNTHETIC-INSTALL-SECRET" not in caplog.text


def test_plugin_hook_failure_is_retryable_without_remount(stub_http_server, monkeypatch, caplog):
    from occ3_sglang import capture
    app = FastAPI()
    stub_http_server.app = app
    plugin._LOADED = False
    def fail():
        raise RuntimeError("SYNTHETIC-HOOK-SECRET")
    monkeypatch.setattr(capture, "register_capture_hook", fail)
    plugin.load()
    assert not plugin._LOADED
    count = len(app.router.routes)
    monkeypatch.setattr(capture, "register_capture_hook", lambda: None)
    plugin.load()
    assert plugin._LOADED and len(app.router.routes) == count
    assert "SYNTHETIC-HOOK-SECRET" not in caplog.text


async def test_capture_deadline_cancels_once_and_retains_uncertain_quota(kv_client, monkeypatch):
    import asyncio
    from occ3_sglang import kv_handles as kv
    client, tm, clock = kv_client
    monkeypatch.setattr(kv, "OPERATION_TIMEOUT_S", 0.02)
    monkeypatch.setattr(kv, "CANCEL_GRACE_S", 0.02)
    monkeypatch.setattr(kv, "MAX_CAPTURES_PER_SESSION", 1)
    closed = asyncio.Event()
    observed = []
    async def hangs(obj, request=None):
        observed.append(obj.rid)
        try:
            await asyncio.Event().wait()
            yield {}
        finally:
            closed.set()
    tm.generate_request = hangs
    body = kv.CaptureBody(**kv_bound(client, text="source"))
    with pytest.raises(kv.ProtocolError) as failure:
        await asyncio.wait_for(kv.capture_source(body), 0.5)
    assert failure.value.status_code == 504 and closed.is_set()
    pending = list(kv.store.captures.values())[0]
    assert pending.state == "UNCERTAIN" and pending.local_task_closed
    assert pending.abort_admitted
    assert [req.rid for req in tm.abort_requests] == observed
    assert all(req.abort_all is False for req in tm.abort_requests)
    clock[0] += 100000
    with kv.store.lock:
        kv.store._sweep()
    assert kv.store.captures[pending.handle] is pending
    with pytest.raises(kv.ProtocolError) as quota:
        await kv.capture_source(body)
    assert quota.value.status_code == 429 and len(observed) == 1
    release = kv.ReleaseHandleBody(**kv_bound(client, capture_handle=pending.handle))
    with pytest.raises(kv.ProtocolError) as uncertain:
        await kv.release_handle(release)
    assert uncertain.value.detail == "cleanup_unconfirmed"


async def test_cancellation_resistant_operation_does_not_hold_route_open_or_reuse_slot(kv_client, monkeypatch):
    import asyncio
    from occ3_sglang import kv_handles as kv
    client, tm, clock = kv_client
    monkeypatch.setattr(kv, "OPERATION_TIMEOUT_S", 0.01)
    monkeypatch.setattr(kv, "CANCEL_GRACE_S", 0.01)
    finish = asyncio.Event()
    cancelled = asyncio.Event()
    async def resistant(obj, request=None):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            await finish.wait()
        yield {"private": "ignored-late-answer"}
    tm.generate_request = resistant
    body = kv.CaptureBody(**kv_bound(client, text="source"))
    try:
        with pytest.raises(kv.ProtocolError) as error:
            await asyncio.wait_for(kv.capture_source(body), 0.5)
        assert error.value.status_code == 504 and cancelled.is_set()
        pending = list(kv.store.captures.values())[0]
        assert pending.state == "UNCERTAIN" and not pending.local_task_closed
        assert pending.operation_task is not None and not pending.operation_task.done()
        assert pending.in_flight
    finally:
        finish.set()
        tasks = [cap.operation_task for cap in kv.store.captures.values()]
        await asyncio.gather(*tasks, return_exceptions=True)
    assert pending.state == "UNCERTAIN" and pending.local_task_closed
    assert kv.store.captures[pending.handle] is pending


async def test_receive_expiry_and_concurrent_release_never_replay(kv_client, monkeypatch):
    import asyncio
    from occ3_sglang import kv_handles as kv
    client, tm, clock = kv_client
    source, sent = kv_capture_and_send(client)
    monkeypatch.setattr(kv, "CANCEL_GRACE_S", 0.01)
    started = asyncio.Event()
    seen = []
    async def hangs(obj, request=None):
        seen.append(obj.rid)
        started.set()
        await asyncio.Event().wait()
        yield {}
    tm.generate_request = hangs
    receive = kv.ReceiveHandleBody(**kv_bound(client, session="receiver",
        channel_handle=sent["channel_handle"], suffix="task", max_new_tokens=64))
    task = asyncio.create_task(kv.receive_handle(receive))
    await asyncio.wait_for(started.wait(), 0.5)
    release = kv.ReleaseHandleBody(**kv_bound(client, capture_handle=source["capture_handle"]))
    with pytest.raises(kv.ProtocolError):
        await kv.release_handle(release)
    with pytest.raises(kv.ProtocolError):
        await kv.receive_handle(receive)
    assert len(seen) == 1
    clock[0] += 601
    with kv.store.lock:
        kv.store._sweep()
    with pytest.raises(kv.ProtocolError) as expired:
        await asyncio.wait_for(task, 0.5)
    assert expired.value.status_code == 504
    pending = kv.store.captures[source["capture_handle"]]
    assert pending.state == "UNCERTAIN"
    assert [req.rid for req in tm.abort_requests] == seen
    with pytest.raises(kv.ProtocolError) as unavailable:
        await kv.receive_handle(receive)
    assert unavailable.value.status_code == 410 and len(seen) == 1


async def test_caller_cancel_propagates_and_burns_source_lease(kv_client, monkeypatch):
    import asyncio
    from occ3_sglang import kv_handles as kv
    client, tm, clock = kv_client
    monkeypatch.setattr(kv, "CANCEL_GRACE_S", 0.01)
    started = asyncio.Event()
    async def hangs(obj, request=None):
        started.set()
        await asyncio.Event().wait()
        yield {}
    tm.generate_request = hangs
    task = asyncio.create_task(kv.capture_source(kv.CaptureBody(**kv_bound(client, text="source"))))
    await asyncio.wait_for(started.wait(), 0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    pending = list(kv.store.captures.values())[0]
    assert pending.state == "UNCERTAIN" and pending.abort_admitted
    assert pending.local_task_closed and len(tm.abort_requests) == 1


async def test_expiry_completion_race_still_admits_exact_abort_once(kv_client):
    from occ3_sglang import kv_handles as kv
    client, tm, clock = kv_client
    source, sent = kv_capture_and_send(client)
    async def raced(obj, request=None):
        # Model the completion/expiry interleaving without any wall-clock polling.
        clock[0] += 601
        with kv.store.lock:
            current = kv.store.captures[source["capture_handle"]]
            kv.store._uncertain(current, "operation_deadline")
        yield {"private": "must-not-publish"}
    tm.generate_request = raced
    body = kv.ReceiveHandleBody(**kv_bound(client, session="receiver",
        channel_handle=sent["channel_handle"], suffix="task", max_new_tokens=64))
    with pytest.raises(kv.ProtocolError) as failure:
        await kv.receive_handle(body)
    assert failure.value.detail == "cleanup_unconfirmed"
    current = kv.store.captures[source["capture_handle"]]
    assert current.state == "UNCERTAIN" and current.abort_admitted
    assert [req.rid for req in tm.abort_requests] == [current.operation_rid]
