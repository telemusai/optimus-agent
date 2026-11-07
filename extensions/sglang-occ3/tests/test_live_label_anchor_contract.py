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
