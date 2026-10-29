"""Frontend real backend with synthetic control transport/journal; NOT GPU proof."""
import asyncio
from dataclasses import asdict
from types import SimpleNamespace as NS
import sys
import time
import types
import pytest

from occ3_sglang.latent_backend import ControlClient, LatentBackend, deliver_control
from occ3_sglang.latent_control_api import BrokerAuthority
from occ3_sglang.latent_types import LatentError, SessionEnrollment
from occ3_sglang.attachment_contract import (AuthenticatedContext, CaptureIntent, InferenceAttachment,
                                            CaptureWorkerDescriptor, ConsumeWorkerDescriptor)
from occ3_sglang.chat_adapter import NativeRequestView
from occ3_sglang.use_receipts import NativeCompletion, SourceToolCall, SourceToolProvenance
from test_latent_types import profile
from test_span_markers import FakeTokenizer, NoEncodeTokenizer


def health(p):
    return dict(runtime_epoch=p.runtime_epoch,runtime_generation=p.runtime_generation,
        model_fingerprint=p.model_fingerprint,method_id=p.method,hidden_dim=p.hidden_dim,
        layer_index=p.layer_index,ready_ranks=8,expected_ranks=8,profile_digest=p.binding_digest,
        tp_layout_digest="b"*64,dtype="bfloat16",source_capture=True,continuous_prefix=True,
        terminal_journal=True,max_capture_bytes=128*1024*1024)


class SyntheticClient:
    def __init__(self,p):
        self.p=p; self.calls=[]; self.record={}
    async def call(self,op,payload,command_id=None):
        self.calls.append((op,payload,command_id))
        if op=="health": return health(self.p)
        if op=="reserve_capture":
            self.record=dict(source_handle=payload["source_handle"],state_generation=payload["state_generation"],
                expires_at_epoch=payload["expires_at_epoch"],state="reserved",revoked=False,source={},terminal={})
        if op=="bind_output": self.record["state"]="ready"
        return self.record
    async def close(self): pass


def fixture():
    p=profile(); auth=BrokerAuthority(p,principal_id="server-principal")
    expiry=int(time.time())+200
    for session in ("source","receiver"):
        auth.remember_session(SessionEnrollment(auth.principal_digest,session,p.model_fingerprint,p.runtime_epoch,9,"domain",expiry))
    client=SyntheticClient(p); backend=LatentBackend(client,p,auth)
    identity=AuthenticatedContext(auth.principal_digest,"source",p.model_id,p.model_fingerprint,p.runtime_epoch,9,time.monotonic()+100)
    return backend,client,identity,expiry


@pytest.mark.asyncio
async def test_source_exact_enrollment_prepare_bind_and_commit():
    backend,client,identity,expiry=fixture(); p=backend.profile
    intent=CaptureIntent(capture_id="capture",source_session_id="source",source_request_id="client-rid",turn_nonce="turn",
        model_fingerprint=p.model_fingerprint,runtime_id=p.runtime_epoch,runtime_generation=9,method=p.method,
        retained_count_limit=8,max_capture_bytes=128*1024*1024,expires_at_epoch=expiry)
    enrollment=await backend.enroll_source(intent,identity,"attempt","native-rid",identity.deadline_monotonic)
    native=NS(rid="native-rid",sampling_params={"max_new_tokens":200},input_ids=[1,2],stream=True)
    view=NativeRequestView(native,200,"s",True,identity.deadline_monotonic)
    reservation=await backend.prepare_source(enrollment,view)
    backend.bind_source(enrollment,reservation,native)
    assert type(native.occ3_worker_descriptor) is CaptureWorkerDescriptor
    assert native.input_ids==[1,2] and native.sampling_params=={"max_new_tokens":200}
    client.record["source"]=dict(capture_commit_id="commit",accepted_span_start=2,accepted_span_end=10,
        retained_count=8,predictor_position_offset=-1,representation="final_layer_prenorm_predictor",
        layer_index=77,hidden_dim=6144,dtype="bfloat16",layout_digest="b"*64,ready_ranks=8,expected_ranks=8,
        accepted_sequence_digest="c"*64,expires_at_epoch=expiry,byte_count=8*6144*2)
    provenance=SourceToolProvenance(tool_calls=(SourceToolCall("tool","occ3_handoff"),))
    completion=NativeCompletion("native-rid","tool_calls","d"*64,50,True)
    evidence=await backend.finish_source(enrollment,reservation,completion,provenance)
    assert evidence.commit.capture_commit_id=="commit" and evidence.bound_output_sha256=="d"*64
    assert client.calls[-1][1]["binding"]["tool_call_id"]=="tool"


@pytest.mark.asyncio
async def test_source_commit_carries_method_metadata_without_type_error():
    # Red test: the worker's method_metadata() has added method_id (and
    # artifact_digest for a calibrated method) to record.source_metadata since
    # wave 445d1cdd4, so production source captures reach finish_source with
    # more keys than SourceCaptureCommit(**record["source"]) accepts. The
    # backend must project the commit, not splat the worker metadata.
    backend,client,identity,expiry=fixture(); p=backend.profile
    intent=CaptureIntent(capture_id="capture",source_session_id="source",source_request_id="client-rid",turn_nonce="turn",
        model_fingerprint=p.model_fingerprint,runtime_id=p.runtime_epoch,runtime_generation=9,method=p.method,
        retained_count_limit=8,max_capture_bytes=128*1024*1024,expires_at_epoch=expiry)
    enrollment=await backend.enroll_source(intent,identity,"attempt","native-rid",identity.deadline_monotonic)
    native=NS(rid="native-rid",sampling_params={"max_new_tokens":200},input_ids=[1,2],stream=True)
    view=NativeRequestView(native,200,"s",True,identity.deadline_monotonic)
    reservation=await backend.prepare_source(enrollment,view)
    backend.bind_source(enrollment,reservation,native)
    client.record["source"]=dict(capture_commit_id="commit",accepted_span_start=2,accepted_span_end=10,
        retained_count=8,predictor_position_offset=-1,representation="final_layer_prenorm_predictor",
        layer_index=77,hidden_dim=6144,dtype="bfloat16",layout_digest="b"*64,ready_ranks=8,expected_ranks=8,
        accepted_sequence_digest="c"*64,expires_at_epoch=expiry,byte_count=8*6144*2,
        method_id=p.method,artifact_digest="e"*64)
    provenance=SourceToolProvenance(tool_calls=())
    completion=NativeCompletion("native-rid","stop","d"*64,50,True)
    evidence=await backend.finish_source(enrollment,reservation,completion,provenance)
    assert evidence.commit.capture_commit_id=="commit" and evidence.bound_output_sha256=="d"*64


@pytest.mark.asyncio
async def test_v2_enroll_resolves_marker_profile_from_the_tokenizer():
    backend,client,identity,expiry=fixture(); p=backend.profile
    client.tm=NS(tokenizer=FakeTokenizer())
    intent=CaptureIntent(capture_id="capture",source_session_id="source",source_request_id="client-rid",turn_nonce="turn",
        model_fingerprint=p.model_fingerprint,runtime_id=p.runtime_epoch,runtime_generation=9,method=p.method,
        retained_count_limit=8,max_capture_bytes=128*1024*1024,expires_at_epoch=expiry,
        schema="occ3.capture.v2",span_policy="content_tail_v2")
    enrollment=await backend.enroll_source(intent,identity,"attempt","native-rid",identity.deadline_monotonic)
    descriptor=backend.attempts["attempt"]["descriptor"]
    assert descriptor.marker_profile=="glm45_glm47_xml_kv.v1"
    assert descriptor.span_policy=="content_tail_v2"
    # The attested marker ids never travel on the descriptor.
    assert not any(isinstance(v,list) for v in asdict(descriptor).values())
    assert enrollment.attempt_id=="attempt"


@pytest.mark.asyncio
async def test_v2_enroll_refuses_an_unattestable_tokenizer():
    backend,client,identity,expiry=fixture(); p=backend.profile
    client.tm=NS(tokenizer=NoEncodeTokenizer())
    intent=CaptureIntent(capture_id="capture",source_session_id="source",source_request_id="client-rid",turn_nonce="turn",
        model_fingerprint=p.model_fingerprint,runtime_id=p.runtime_epoch,runtime_generation=9,method=p.method,
        retained_count_limit=8,max_capture_bytes=128*1024*1024,expires_at_epoch=expiry,
        schema="occ3.capture.v2",span_policy="content_tail_v2")
    with pytest.raises(LatentError) as caught:
        await backend.enroll_source(intent,identity,"attempt","native-rid",identity.deadline_monotonic)
    assert caught.value.code=="profile" and not backend.attempts


@pytest.mark.asyncio
async def test_receiver_binds_only_descriptor_and_requires_terminal_journal():
    backend,client,identity,expiry=fixture(); p=backend.profile
    identity=AuthenticatedContext(identity.principal_digest,"receiver",p.model_id,p.model_fingerprint,p.runtime_epoch,9,identity.deadline_monotonic)
    attachment=InferenceAttachment(1,"nonce","message","connector","receiver",p.model_fingerprint,p.runtime_epoch,9,"handle",expiry,p.method)
    client.record=dict(source_handle="handle",state_generation=7,expires_at_epoch=expiry,state="published",revoked=False,
        source=dict(capture_commit_id="commit",expected_ranks=8,retained_count=8,hidden_dim=6144,dtype="bfloat16",layout_digest="b"*64),terminal={})
    native=NS(rid="native-rid",input_ids=[1],sampling_params={"max_new_tokens":100},stream=True)
    reservation=await backend.reserve(attachment,identity,"consume",NativeRequestView(native,100,"s",True,identity.deadline_monotonic))
    backend.bind(reservation,native)
    assert type(native.occ3_worker_descriptor) is ConsumeWorkerDescriptor and native.input_ids==[1]
    completion=NativeCompletion("native-rid","stop","d"*64,1,True)
    with pytest.raises(LatentError): await backend.finish(reservation,completion)
    client.record["state"]="terminal"
    client.record["terminal"]=dict(native_rid="native-rid",injection_commit_id="injected",completed=True,
        gpu_resident=True,num_retractions=0,private_cache_namespace=True,injected_prefix_count=8,ready_ranks=8,expected_ranks=8)
    assert (await backend.finish(reservation,completion)).injection_commit_id=="injected"


@pytest.mark.asyncio
async def test_control_client_exact_id_delivery_and_detach(monkeypatch):
    module=types.ModuleType("sglang.srt.managers.io_struct")
    module.Occ3ControlReq=lambda **kw:NS(**kw)
    monkeypatch.setitem(sys.modules,"sglang.srt.managers.io_struct",module)
    tm=NS(auto_create_handle_loop=lambda:None)
    async def dispatch(request):
        deliver_control(tm,dict(schema="occ3.control.result.v1",command_id=request.rid,ok=True,result={"x":1},error=""))
    tm._async_dispatch_to_scheduler=dispatch
    client=ControlClient(tm)
    assert await client.call("health",{})=={"x":1}
    assert not client.pending
    await client.close()
    assert not hasattr(tm,"_occ3_control_client")


@pytest.mark.asyncio
async def test_cancellation_does_not_resend(monkeypatch):
    module=types.ModuleType("sglang.srt.managers.io_struct"); module.Occ3ControlReq=lambda **kw:NS(**kw)
    monkeypatch.setitem(sys.modules,"sglang.srt.managers.io_struct",module)
    submitted=asyncio.Event(); calls=[]
    async def dispatch(request): calls.append(request); submitted.set()
    tm=NS(auto_create_handle_loop=lambda:None,_async_dispatch_to_scheduler=dispatch)
    client=ControlClient(tm)
    task=asyncio.create_task(client.call("health",{})); await submitted.wait(); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert len(calls)==1 and not client.pending
    await client.close()


@pytest.mark.asyncio
async def test_canceled_control_remains_uncertain_until_valid_late_reply(monkeypatch):
    module=types.ModuleType("sglang.srt.managers.io_struct"); module.Occ3ControlReq=lambda **kw:NS(**kw)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    submitted=asyncio.Event(); calls=[]
    async def dispatch(request): calls.append(request); submitted.set()
    tm=NS(auto_create_handle_loop=lambda:None,_async_dispatch_to_scheduler=dispatch)
    client=ControlClient(tm,max_pending=1)
    task=asyncio.create_task(client.call("health",{},command_id="first"))
    await submitted.wait(); task.cancel()
    with pytest.raises(asyncio.CancelledError): await task
    assert client.has_uncertain_control and not client.pending
    with pytest.raises(LatentError): await client.call("health",{})
    await client.close()
    assert tm._occ3_control_client is client and client.has_uncertain_control
    deliver_control(tm,dict(schema="occ3.control.result.v1",command_id="other",ok=True,result={},error=""))
    assert client.has_uncertain_control
    with pytest.raises(LatentError): deliver_control(tm,{"command_id":"first"})
    assert client.has_uncertain_control
    deliver_control(tm,dict(schema="occ3.control.result.v1",command_id="first",ok=True,result={},error=""))
    assert not client.has_uncertain_control and not hasattr(tm,"_occ3_control_client")


@pytest.mark.asyncio
async def test_partial_dispatch_failure_keeps_control_uncertainty(monkeypatch):
    module=types.ModuleType("sglang.srt.managers.io_struct"); module.Occ3ControlReq=lambda **kw:NS(**kw)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    async def dispatch(request): raise RuntimeError("transport may have sent")
    tm=NS(auto_create_handle_loop=lambda:None,_async_dispatch_to_scheduler=dispatch)
    client=ControlClient(tm)
    with pytest.raises(RuntimeError): await client.call("health",{},command_id="partial")
    await client.close()
    assert client.has_uncertain_control and tm._occ3_control_client is client
    deliver_control(tm,dict(schema="occ3.control.result.v1",command_id="partial",ok=False,result={},error="unavailable"))
    assert not client.has_uncertain_control


@pytest.mark.asyncio
async def test_pre_dispatch_failure_has_no_false_uncertainty(monkeypatch):
    module=types.ModuleType("sglang.srt.managers.io_struct"); module.Occ3ControlReq=lambda **kw:NS(**kw)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    def fail(): raise RuntimeError("handle loop not created")
    tm=NS(auto_create_handle_loop=fail)
    client=ControlClient(tm)
    with pytest.raises(RuntimeError): await client.call("health",{})
    await client.close()
    assert not client.has_uncertain_control and not hasattr(tm,"_occ3_control_client")


@pytest.mark.asyncio
async def test_initial_send_timeout_is_bounded_and_retains_resistant_operation(monkeypatch):
    module=types.ModuleType("sglang.srt.managers.io_struct"); module.Occ3ControlReq=lambda **kw:NS(**kw)
    monkeypatch.setitem(sys.modules,module.__name__,module)
    release=asyncio.Event(); canceled=asyncio.Event()
    async def dispatch(request):
        while not release.is_set():
            try: await release.wait()
            except asyncio.CancelledError: canceled.set()
    tm=NS(auto_create_handle_loop=lambda:None,_async_dispatch_to_scheduler=dispatch)
    client=ControlClient(tm,timeout_seconds=0.01,max_pending=1)
    with pytest.raises(asyncio.TimeoutError): await client.call("health",{},command_id="stalled")
    await canceled.wait()
    assert client.has_uncertain_control and not client.pending and client._operations
    await client.close()
    deliver_control(tm,dict(schema="occ3.control.result.v1",command_id="stalled",ok=True,result={},error=""))
    assert client.has_uncertain_control and tm._occ3_control_client is client
    tasks=list(client._operations.values())
    release.set()
    await asyncio.gather(*tasks,return_exceptions=True)
    assert not client.has_uncertain_control
    assert not hasattr(tm,"_occ3_control_client")


@pytest.mark.asyncio
async def test_backend_health_epoch_mapping_uses_actual_worker_projection():
    # Invoke production projection on a metadata fixture, not GPU readiness proof.
    from occ3_sglang.latent_worker import WorkerRuntime
    backend,client,identity,expiry=fixture()
    worker=object.__new__(WorkerRuntime)
    worker.profile=backend.profile; worker.initialized=True; worker.poisoned=False
    worker.layout_digest="b"*64; worker.dtype="bfloat16"
    actual=worker.health()
    original_call=client.call
    async def call(op,payload,command_id=None):
        if op=="health": return dict(actual)
        return await original_call(op,payload,command_id)
    client.call=call
    result=await backend.refresh_readiness()
    assert result["runtime_epoch"]==identity.runtime_id==backend.profile.runtime_epoch
    assert result["runtime_generation"]==identity.runtime_generation
    assert result["method_id"]=="statebridge.glm53.final-prenorm-predictor.continuous-prefix.v1"
    actual["runtime_epoch"]="deployment-label-not-fresh-epoch"
    with pytest.raises(LatentError,match="identity"): await backend.refresh_readiness()
    actual["runtime_epoch"]=backend.profile.runtime_epoch
    actual["runtime_generation"]+=1
    with pytest.raises(LatentError,match="identity"): await backend.refresh_readiness()


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", [("runtime_id","deployment-label"),("runtime_generation",10),
    ("method","statebridge-prefix"),("model_fingerprint","different")])
async def test_receiver_rejects_epoch_method_alias_before_control(field,value):
    from dataclasses import replace
    backend,client,identity,expiry=fixture(); p=backend.profile
    identity=AuthenticatedContext(identity.principal_digest,"receiver",p.model_id,p.model_fingerprint,p.runtime_epoch,9,identity.deadline_monotonic)
    attachment=InferenceAttachment(1,"nonce","message","connector","receiver",p.model_fingerprint,p.runtime_epoch,9,"handle",expiry,p.method)
    attachment=replace(attachment,**{field:value})
    native=NS(rid="rid",max_new_tokens=10)
    with pytest.raises(LatentError,match="identity"):
        await backend.reserve(attachment,identity,"attempt",NativeRequestView(native,10,"s",True,identity.deadline_monotonic))
    assert not client.calls
