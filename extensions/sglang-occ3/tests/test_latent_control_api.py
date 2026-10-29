"""Real FastAPI routes + real authority with synthetic control backend only."""
import json
import time
from types import SimpleNamespace as NS
import pytest
from fastapi import FastAPI
import httpx
from starlette.requests import Request

from occ3_sglang.latent_control_api import BrokerAuthority, create_router
from occ3_sglang.latent_types import LatentError, SessionEnrollment, bounded_payload
from occ3_sglang.attachment_contract import (AttachmentError, _SOURCE_RECEIPT_V1_FIELDS)
from occ3_sglang import startup_state
from test_latent_types import profile
from test_latent_backend import SyntheticClient, health


def app_fixture(verified=True,refresh_error=None):
    p=profile(); authority=BrokerAuthority(p,principal_id="server-owned")
    client=SyntheticClient(p); client.refreshes=[]
    async def refresh():
        client.refreshes.append(time.monotonic())
        if refresh_error is not None: raise LatentError(refresh_error)
        return health(p)
    backend=NS(client=client,refresh_readiness=refresh,profile=p)
    app=FastAPI()
    @app.middleware("http")
    async def auth(request,call_next):
        authority.clear_verified(request)
        if verified:
            authority.mark_verified(request,"server-owned",deadline_monotonic=time.monotonic()+300)
        return await call_next(request)
    app.include_router(create_router(backend,authority))
    app.include_router(startup_state.create_control_router())
    startup_state.record_native_ready(attempt="fixture0", installation=NS(backend=backend, authority=authority))
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://test"),authority,client


def enroll_body():
    return dict(schema="occ3.sessions.enroll.v1",operation_id="operation",session_id="source",
        model_fingerprint="model-hash",runtime_id="epoch",runtime_generation=9,access_domain="family",
        expires_at_epoch=int(time.time())+100)


IDENTITY_REQUEST={"schema":"occ3.runtime.identity.request.v1"}
IDENTITY_FIELDS={"schema","runtime_id","runtime_generation","model_fingerprint","method_id",
    "tp_layout_digest","layer_index","hidden_dim","dtype","expected_ranks","ready_ranks",
    "profile_digest","source_capture","continuous_prefix","terminal_journal",
    "server_time_epoch","max_ttl_seconds","max_capture_bytes"}


@pytest.mark.asyncio
async def test_identity_route_returns_current_runtime_tuple():
    client,authority,transport=app_fixture()
    p=profile(); now=time.time()
    response=await client.post("/occ3/v1/native/runtime/identity",json=IDENTITY_REQUEST)
    assert response.status_code==200, response.text
    assert response.headers["cache-control"]=="no-store"
    result=response.json()
    assert set(result)==IDENTITY_FIELDS
    assert result["schema"]=="occ3.runtime.identity.v1"
    assert result["runtime_id"]==p.runtime_epoch and result["runtime_generation"]==p.runtime_generation
    assert result["model_fingerprint"]==p.model_fingerprint and result["method_id"]==p.method
    assert result["tp_layout_digest"]==health(p)["tp_layout_digest"]
    assert (result["layer_index"],result["hidden_dim"],result["dtype"])==(p.layer_index,p.hidden_dim,"bfloat16")
    assert result["expected_ranks"]==result["ready_ranks"]==8
    assert result["profile_digest"]==p.binding_digest
    assert result["source_capture"] and result["continuous_prefix"] and result["terminal_journal"]
    assert now-5<=result["server_time_epoch"]<=time.time()+5
    assert result["max_ttl_seconds"]==p.max_ttl_seconds
    assert result["max_capture_bytes"]==128*1024*1024  # server wire cap advertised
    assert len(transport.refreshes)==1  # Fresh ordered health per request; never cached.


@pytest.mark.asyncio
async def test_identity_route_requires_verified_principal():
    client,authority,transport=app_fixture(False)
    response=await client.post("/occ3/v1/native/runtime/identity",json=IDENTITY_REQUEST)
    assert response.status_code==401
    assert not transport.calls and not transport.refreshes  # Refused before any backend call.


@pytest.mark.parametrize("body", [b'{"schema":"x"}',
    b'{"schema":"occ3.runtime.identity.request.v1","extra":1}',
    b'{"schema":"a","schema":"b"}',
    b'{"schema":"x","padding":"'+b'a'*17000+b'"}'])
@pytest.mark.asyncio
async def test_identity_route_strict_bodies(body):
    client,_,_=app_fixture()
    response=await client.post("/occ3/v1/native/runtime/identity",content=body)
    assert response.status_code in (400,429)
    assert "runtime_epoch" not in response.text and "readiness" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("code,status",[("unavailable",503),("identity",401),("quota",429)])
async def test_identity_route_maps_backend_errors(code,status):
    client,authority,transport=app_fixture(refresh_error=code)
    response=await client.post("/occ3/v1/native/runtime/identity",json=IDENTITY_REQUEST)
    assert response.status_code==status and response.json()=={"error":{"code":code}}
    assert transport.refreshes  # The shared execute() mapping is exercised, nothing else.


@pytest.mark.asyncio
async def test_identity_route_response_is_metadata_only():
    client,_,_=app_fixture()
    response=await client.post("/occ3/v1/native/runtime/identity",json=IDENTITY_REQUEST)
    assert response.status_code==200 and response.headers["cache-control"]=="no-store"
    for leaked in ("handle","session","nonce","principal","turn","attempt","binding_digest","tokens","tensor"):
        assert leaked not in response.text
    bounded_payload(response.json())  # Strict-payload round trip: bounded, no extras.


@pytest.mark.asyncio
async def test_identity_route_rejects_non_post():
    client,_,_=app_fixture()
    for method in (client.get,client.put,client.delete):
        response=await method("/occ3/v1/native/runtime/identity")
        assert response.status_code==405


@pytest.mark.asyncio
async def test_unverified_request_cannot_enroll_or_use_labels():
    client,authority,transport=app_fixture(False)
    response=await client.post("/occ3/v1/native/sessions/enroll",json=enroll_body(),headers={"x-principal":"server-owned"})
    assert response.status_code==401 and not transport.calls


@pytest.mark.asyncio
async def test_enrollment_is_worker_checked_before_authority_table():
    client,authority,transport=app_fixture()
    # Synthetic transport supplies explicit readiness, not GPU proof.
    transport.record=health(profile())
    response=await client.post("/occ3/v1/native/sessions/enroll",json=enroll_body())
    assert response.status_code==200, response.text
    assert response.json()["principal_digest"]==authority.principal_digest
    assert authority.check_session(authority.principal_digest,"source")
    assert transport.calls[0][0]=="enroll_session"


@pytest.mark.parametrize("body", [b'{"schema":"x","schema":"y"}', b'{}', b'{"x":"'+b'a'*17000+b'"}'])
@pytest.mark.asyncio
async def test_bounded_strict_control(body):
    client,_,_=app_fixture()
    response=await client.post("/occ3/v1/native/sessions/enroll",content=body)
    assert response.status_code in (400,429)
    assert "source_handle" not in response.text


def test_private_marker_cannot_be_prepopulated():
    authority=BrokerAuthority(profile(),principal_id="server-owned")
    request=Request({"type":"http","state":{"_occ3_verified_broker":("wrong",authority.principal_digest,9999)}})
    with pytest.raises(LatentError): authority.verified(request)
    authority.clear_verified(request)
    authority.mark_verified(request,"server-owned")
    assert authority.verified(request)[0]==authority.principal_digest
    authority.clear_verified(request)
    with pytest.raises(LatentError): authority.verified(request)


@pytest.mark.asyncio
async def test_authorizer_requires_enrollment_model_and_deadline():
    p=profile(); authority=BrokerAuthority(p,principal_id="server-owned")
    expiry=int(time.time())+100
    authority.remember_session(SessionEnrollment(authority.principal_digest,"source",p.model_fingerprint,p.runtime_epoch,9,"domain",expiry))
    body={"model":"glm","occ3_capture":dict(schema="occ3.capture.v1",capture_id="capture",source_session_id="source",
        source_request_id="client",turn_nonce="turn",model_fingerprint=p.model_fingerprint,runtime_id="epoch",
        runtime_generation=9,method=p.method,retained_count_limit=8,max_capture_bytes=128*1024*1024,
        expires_at_epoch=expiry,span_policy="whole_completed_response_tail_v1")}
    request=Request({"type":"http","state":{}}); request._body=json.dumps(body).encode()
    authority.mark_verified(request,"server-owned")
    with pytest.raises(AttachmentError,match="deadline_unavailable"): await authority.authorize(request)
    authority.mark_verified(request,"server-owned",deadline_monotonic=time.monotonic()+300)
    identity=await authority.authorize(request)
    assert identity.session_id=="source" and identity.deadline_monotonic < time.monotonic()+110


async def build_native_enrollment_fixture():
    """Production projections on explicit metadata-only/rank/transport fixtures.

    No WorkerRuntime constructor, Torch import, checkpoint or GPU work occurs.
    The emitted JSON is conformance evidence, never a usable runtime lease.
    """
    from dataclasses import asdict
    from occ3_sglang.latent_backend import LatentBackend
    from occ3_sglang.latent_registry import LatentRegistry
    from occ3_sglang.latent_types import RuntimeProfile, control_request
    from occ3_sglang.latent_worker import WorkerRuntime
    p=RuntimeProfile(runtime_epoch="fixture-boot-epoch-001",runtime_generation=7,
        model_id="glm-5.3",model_fingerprint="sha256:db2c8830820da8bd3370bbfacf0799e33085ef7b8b1ca27840b88e3e8101b732")
    worker=object.__new__(WorkerRuntime)
    worker.profile=p; worker.initialized=True; worker.poisoned=False; worker.commands={}
    worker.layout_digest="b"*64; worker.dtype="bfloat16"
    worker.registry=LatentRegistry(p,epoch=lambda:2000000000.0,monotonic=lambda:100.0)
    worker.gather=lambda value:[value]*8  # Explicit simulated ranks, not NCCL/GLOO proof.
    class LocalControl:
        def __init__(self): self.sequence=0
        async def call(self,op,payload,command_id=None):
            self.sequence+=1
            result=worker.control(control_request(command_id or "fixture-"+str(self.sequence),op,payload))
            if not result["ok"]: raise LatentError(result["error"])
            return result["result"]
    authority=BrokerAuthority(p,principal_id="fixture-server-owned-principal")
    backend=LatentBackend(LocalControl(),p,authority)
    app=FastAPI()
    @app.middleware("http")
    async def verified_fixture(request,call_next):
        authority.clear_verified(request)
        authority.mark_verified(request,"fixture-server-owned-principal",deadline_monotonic=time.monotonic()+300)
        return await call_next(request)
    app.include_router(create_router(backend,authority))
    body=dict(schema="occ3.sessions.enroll.v1",operation_id="fixture-operation",session_id="fixture-source",
        model_fingerprint=p.model_fingerprint,runtime_id=p.runtime_epoch,runtime_generation=p.runtime_generation,
        access_domain="fixture-family",expires_at_epoch=2000000100)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app),base_url="http://fixture.invalid") as client:
        response=await client.post("/occ3/v1/native/sessions/enroll",json=body)
        assert response.status_code==200,response.text
        positive=response.json()
        negatives=[]
        for field,value in (("runtime_id","fixture-boot-epoch-restarted"),("runtime_generation",8)):
            changed={**body,"operation_id":"negative-"+field,field:value}
            rejected=await client.post("/occ3/v1/native/sessions/enroll",json=changed)
            assert rejected.status_code==401
            negatives.append(dict(name="request-"+field,request=changed,status=rejected.status_code,response=rejected.json()))
    # Independently tampered wrapper/nested fields must be rejected by native,
    # even when the other epoch field still equals the trusted expected value.
    response_negatives=[]
    for field in ("runtime_id","readiness.runtime_epoch","runtime_generation","readiness.runtime_generation"):
        changed=json.loads(json.dumps(positive))
        if field.startswith("readiness."):
            key=field.split(".")[1]; changed["readiness"][key]="fixture-boot-epoch-restarted" if key=="runtime_epoch" else 8
        else: changed[field]="fixture-boot-epoch-restarted" if field=="runtime_id" else 8
        response_negatives.append(dict(name=field,response=changed,native_must_reject=True))
    from occ3_sglang.attachment_contract import CaptureIntent, SourceCaptureReceipt
    unchanged=dict(intent_schema="occ3.capture.v1",intent_fields=list(CaptureIntent.__dataclass_fields__),
        receipt_schema="occ3.source.v1",receipt_fields=list(SourceCaptureReceipt.__dataclass_fields__),
        receiver_wire_changed=False)
    return dict(schema="occ3.offline.native-enrollment-fixture.v1",fixture_kind="metadata-only-simulated-ranks-and-transport",
        gpu_validated=False,fixture_epoch_seconds=2000000000,
        expected_runtime_epoch=p.runtime_epoch,expected_runtime_generation=p.runtime_generation,
        expected_method=p.method,profile=asdict(p),valid=dict(request=body,status=200,response=positive),
        unchanged_source_contract=unchanged,request_negatives=negatives,response_negatives=response_negatives)


@pytest.mark.asyncio
async def test_real_backend_projection_fixture_epoch_and_generation_are_not_aliases():
    result=await build_native_enrollment_fixture()
    public=result["valid"]["response"]
    assert public["runtime_id"]==public["readiness"]["runtime_epoch"]==result["expected_runtime_epoch"]
    assert public["runtime_generation"]==public["readiness"]["runtime_generation"]==7
    assert public["readiness"]["method_id"]==result["expected_method"]
    assert len(result["request_negatives"])==2 and len(result["response_negatives"])==4
    assert result["gpu_validated"] is False
    assert len(result["unchanged_source_contract"]["intent_fields"])==13
    # A v1 wire carries exactly the 21 v1 fields; the receipt type also carries
    # the schema-gated v2 span block, so pin the v1 subset, not the total count.
    assert set(_SOURCE_RECEIPT_V1_FIELDS) <= set(result["unchanged_source_contract"]["receipt_fields"])
    assert "runtime_id" not in result["unchanged_source_contract"]["receipt_fields"]
