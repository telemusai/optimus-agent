"""Request selection, resolved IPC and frontend evidence for live label anchors."""
from dataclasses import asdict, replace
import pytest

from occ3_sglang.attachment_contract import (
    AttachmentError, CaptureIntent, canonical, normalize_worker_descriptor,
)
from occ3_sglang.latent_types import LatentError, decode_descriptor
from test_latent_types import capture
from test_latent_backend import fixture


def intent_for(backend, expiry, **changes):
    p = backend.profile
    return CaptureIntent(capture_id="capture", source_session_id="source",
        source_request_id="client-rid", turn_nonce="turn",
        model_fingerprint=p.model_fingerprint, runtime_id=p.runtime_epoch,
        runtime_generation=9, method=p.method, retained_count_limit=8,
        max_capture_bytes=128*1024*1024, expires_at_epoch=expiry, **changes)


@pytest.mark.parametrize("alpha,expected", [
    (None, "5de0308e3e394cd7e5275a5f7c35704853935b63b3eac78dec6617f123b98065"),
    (0.0, "4eb17b21225a458fe37d74ff24638047d183fbbd1124ddda8b51c7fb84d68dae"),
    (0.3, "b85e8d6b733560ce31d25a6a79b9d0aa75ebfa28c65c3b6b68a58d0bdf9b7042"),
    (0.7, "39be23526a02271d94083bcab9296ffff4d627ccb39e8b196b0ce6ca75ccf402"),
    (1.0, "87d087f82898938d6c1dbd59601928b932bb1b9c9c087e2642d57f9edd01b1b0"),
])
def test_capture_intent_matches_rust_canonical_digest_vectors(alpha, expected):
    # Pin the same wire fixture as crates/pi-ai/tests/source_capture.rs.
    intent = CaptureIntent(schema="occ3.capture.v1", capture_id="capture-1",
        source_session_id="source-1", source_request_id="client-request-1",
        turn_nonce="turn-1", model_fingerprint="a" * 64, runtime_id="runtime-1",
        runtime_generation=7, method="statebridge-prenorm-v1",
        retained_count_limit=128, max_capture_bytes=4096,
        expires_at_epoch=2000000000, span_policy="whole_completed_response_tail_v1",
        label_anchor_alpha=alpha)
    assert intent.binding_digest == expected


def test_omission_retains_legacy_wire_and_zero_is_explicit():
    backend, _, _, expiry = fixture()
    intent = intent_for(backend, expiry)
    legacy = asdict(intent)
    legacy.pop("capture_mode")
    legacy.pop("label_anchor_alpha")
    assert canonical(intent.wire()) == canonical(legacy)
    assert CaptureIntent.parse(legacy).label_anchor_alpha is None
    explicit = replace(intent, label_anchor_alpha=0.0)
    assert explicit.wire()["label_anchor_alpha"] == 0.0
    assert explicit.binding_digest != intent.binding_digest


@pytest.mark.parametrize("alpha", [0, .3, .5, .7, 1])
def test_allowed_intent_and_resolved_descriptor(alpha):
    backend, _, _, expiry = fixture()
    intent = intent_for(backend, expiry, label_anchor_alpha=alpha)
    assert CaptureIntent.parse(intent.wire()).label_anchor_alpha == alpha
    descriptor = capture(label_anchor_alpha=alpha)
    assert type(descriptor.label_anchor_alpha) is float
    assert normalize_worker_descriptor(asdict(descriptor)) == descriptor


@pytest.mark.parametrize("alpha", [None, True, False, "0.3", [], {}, .2,
                                      float("inf"), float("nan")])
def test_explicit_invalid_request_and_descriptor_rejected(alpha):
    backend, _, _, expiry = fixture()
    wire = intent_for(backend, expiry).wire()
    wire["label_anchor_alpha"] = alpha
    with pytest.raises(AttachmentError):
        CaptureIntent.parse(wire)
    with pytest.raises(AttachmentError):
        capture(label_anchor_alpha=alpha)


def test_legacy_internal_descriptor_defaults_to_zero():
    wire = asdict(capture())
    wire.pop("label_anchor_alpha")
    assert normalize_worker_descriptor(wire).label_anchor_alpha == 0.0
    assert decode_descriptor(wire).label_anchor_alpha == 0.0


@pytest.mark.asyncio
@pytest.mark.parametrize("selection,expected", [(None,.7), (0.,0.), (.3,.3), (.5,.5), (.7,.7), (1.,1.)])
async def test_backend_resolves_before_dispatch(selection, expected):
    backend, client, identity, expiry = fixture()
    backend.profile = replace(backend.profile, label_anchor_alpha_allowed=(0., .3, .5, .7, 1.),
                              label_anchor_alpha_default=.7)
    client.p = backend.profile
    intent = intent_for(backend, expiry, label_anchor_alpha=selection)
    await backend.enroll_source(intent, identity, "attempt", "native-rid", identity.deadline_monotonic)
    assert backend.attempts["attempt"]["descriptor"].label_anchor_alpha == expected
    assert client.calls[-1][1]["descriptor"]["label_anchor_alpha"] == expected


@pytest.mark.asyncio
async def test_profile_rejection_precedes_any_control_dispatch(monkeypatch):
    backend, client, identity, expiry = fixture()
    def refuse(self, value):
        raise LatentError("profile")
    monkeypatch.setattr(type(backend.profile), "resolve_label_anchor", refuse)
    with pytest.raises(LatentError):
        await backend.enroll_source(intent_for(backend, expiry, label_anchor_alpha=.3),
            identity, "attempt", "native-rid", identity.deadline_monotonic)
    assert client.calls == [] and backend.attempts == {}


def test_registry_validates_selected_alpha_before_reserving(monkeypatch):
    from test_latent_registry import registry
    r, _ = registry()
    descriptor = capture(label_anchor_alpha=.3)
    seen = []
    def refuse(self, value):
        seen.append(value)
        raise LatentError("profile")
    monkeypatch.setattr(type(r.profile), "resolve_label_anchor", refuse)
    with pytest.raises(LatentError):
        r.reserve_capture(descriptor, "handle", 123, 400,
                          128*1024*1024, 4096)
    assert seen == [.3] and not r.attempts


@pytest.mark.asyncio
async def test_out_of_attested_set_rejected_before_dispatch():
    backend, client, identity, expiry = fixture()
    backend.profile = replace(backend.profile, label_anchor_alpha_allowed=(0., .5),
                              label_anchor_alpha_default=0.)
    with pytest.raises(LatentError):
        await backend.enroll_source(intent_for(backend, expiry, label_anchor_alpha=.3),
            identity, "attempt", "native-rid", identity.deadline_monotonic)
    assert client.calls == [] and backend.attempts == {}
