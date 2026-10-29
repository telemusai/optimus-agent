"""Method dispatch: one solver per registered method id, never a substitute."""
from __future__ import annotations

import inspect
import os
from dataclasses import replace
from pathlib import Path

import pytest

from occ3_sglang import alignment_dispatch, latent_types, statebridge, statebridge_calibrated
from occ3_sglang.latent_types import LatentError, RuntimeProfile
from occ3_sglang.native_profile import CALIBRATED_METHOD, PER_MESSAGE_METHOD

DIM, ROWS = 6144, 128
FINGERPRINT = "sha256:" + "a" * 64
# An absolute path that never has to exist: these cases refuse before any read.
PLACEHOLDER = os.path.join(os.path.abspath("."), "placeholder.occ3cal")


def _provenance(**changes):
    values = dict(layer_index=77, model_fingerprint=FINGERPRINT,
                  checkpoint_revision="rev-1", tokenizer_fingerprint="sha256:" + "b" * 64,
                  embedding_id="glm53-tp8-vocab",
                  capture_method_id="glm53.final-prenorm.capture.v1")
    values.update(changes)
    return statebridge_calibrated.CalibrationProvenance(**values)


def _corpus(dim=DIM, rows=ROWS, seed=0):
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(seed)
    rank = 8
    left = torch.randn(rank, dim, generator=generator) / rank ** 0.5
    right = torch.randn(rank, dim, generator=generator) / rank ** 0.5
    factor = torch.randn(rows, rank, generator=generator)
    hidden = 0.95 ** 0.5 * (factor @ left) + 0.05 ** 0.5 * torch.randn(rows, dim, generator=generator)
    embeddings = 0.95 ** 0.5 * (factor @ right) + 0.05 ** 0.5 * torch.randn(rows, dim, generator=generator)
    return hidden, embeddings


def _artifact(seed=2):
    hidden, embeddings = _corpus(seed=seed)
    partition = {"hidden": hidden, "embeddings": embeddings,
                 "row_turn_ids": ("turn-0",) * rows_for(hidden),
                 "turn_ids": ("turn-0",), "manifest_digest": ""}
    return statebridge_calibrated.fit_map(partition, {"rho": 0.02}, provenance=_provenance())


def rows_for(hidden):
    return int(hidden.shape[0])


@pytest.fixture(scope="module")
def calibrated_artifact():
    return _artifact(seed=0)


@pytest.fixture(scope="module")
def artifact_file(tmp_path_factory, calibrated_artifact):
    path = tmp_path_factory.mktemp("frozen") / "calibration.occ3cal"
    path.write_bytes(statebridge_calibrated.serialize_artifact(calibrated_artifact))
    return path


def _fields(**changes):
    values = dict(runtime_epoch="epoch", runtime_generation=9,
                  model_fingerprint=FINGERPRINT, model_id="glm-5.3")
    values.update(changes)
    return values


def _message_profile():
    return RuntimeProfile(**_fields())


def _calibrated_profile(path, digest):
    return RuntimeProfile(**_fields(method=CALIBRATED_METHOD,
                                    calibration_path=str(path), calibration_digest=digest))


def test_per_message_profile_dispatches_to_the_shipped_solver(monkeypatch):
    torch = pytest.importorskip("torch")
    calls, apply_calls = [], []
    original = statebridge.align_hidden_sequence
    monkeypatch.setattr(statebridge, "align_hidden_sequence",
                        lambda *args, **kwargs: calls.append(args) or original(*args, **kwargs))
    monkeypatch.setattr(statebridge_calibrated, "apply_map",
                        lambda *args, **kwargs: apply_calls.append(args))
    hidden = torch.randn(8, DIM)
    embeddings = torch.randn(8, DIM)
    aligner = alignment_dispatch.aligner_for(_message_profile())
    prefix, info = aligner(hidden, embeddings, 1.0)
    assert len(calls) == 1 and apply_calls == []
    assert isinstance(info, statebridge.AlignmentInfo) and info.hidden_dim == DIM
    assert prefix.shape == hidden.shape


def test_calibrated_profile_dispatches_to_the_frozen_transform(monkeypatch):
    torch = pytest.importorskip("torch")
    apply_calls, shipped_calls = [], []
    original = statebridge_calibrated.apply_map
    monkeypatch.setattr(statebridge_calibrated, "apply_map",
                        lambda *args, **kwargs: apply_calls.append(args) or original(*args, **kwargs))
    monkeypatch.setattr(statebridge, "align_hidden_sequence",
                        lambda *args, **kwargs: shipped_calls.append(args))
    artifact = _artifact(seed=1)
    # The profile pins the digest of the artifact it is bound to, so the pair is
    # accepted even though the path is never read on this branch.
    profile = _calibrated_profile(PLACEHOLDER, artifact.content_digest)
    aligner = alignment_dispatch.aligner_for(profile, artifact)
    prefix, info = aligner(torch.randn(8, DIM), torch.randn(8, DIM), 1.0)
    assert len(apply_calls) == 1 and shipped_calls == []
    assert isinstance(info, alignment_dispatch.CalibratedAlignmentInfo)
    assert info.method == CALIBRATED_METHOD and info.hidden_dim == DIM
    assert info.artifact_digest == artifact.content_digest
    assert info.artifact_digest == statebridge_calibrated.artifact_digest(artifact)
    assert prefix.shape == (8, DIM)
    assert aligner.serving is None  # fit default is N0: raw served rows


def test_missing_artifact_refuses_and_never_uses_the_shipped_solver(monkeypatch):
    shipped_calls = []
    monkeypatch.setattr(statebridge, "align_hidden_sequence",
                        lambda *args, **kwargs: shipped_calls.append(args))
    profile = _calibrated_profile(PLACEHOLDER, "c" * 64)
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.aligner_for(profile)
    assert caught.value.code == "unavailable"
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.aligner_for(profile, None)
    assert caught.value.code == "unavailable"
    assert shipped_calls == []


def test_artifact_for_another_method_id_refuses(calibrated_artifact):
    profile = _calibrated_profile(PLACEHOLDER, "c" * 64)
    foreign = replace(calibrated_artifact, method_id=PER_MESSAGE_METHOD)
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.aligner_for(profile, foreign)
    assert caught.value.code == "identity"


def test_artifact_with_another_hidden_dim_refuses(calibrated_artifact):
    profile = _calibrated_profile(PLACEHOLDER, "c" * 64)
    narrower = replace(calibrated_artifact,
                       hidden_dim=32,
                       mean_hidden=calibrated_artifact.mean_hidden[:, :32].contiguous(),
                       mean_embedding=calibrated_artifact.mean_embedding[:, :32].contiguous(),
                       hidden_residual=calibrated_artifact.hidden_residual[:, :32].contiguous(),
                       embedding_residual=calibrated_artifact.embedding_residual[:, :32].contiguous())
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.aligner_for(profile, narrower)
    assert caught.value.code == "profile"


def test_a_calibrated_refusal_never_degrades_to_per_message_output(monkeypatch):
    torch = pytest.importorskip("torch")
    shipped_calls = []
    monkeypatch.setattr(statebridge, "align_hidden_sequence",
                        lambda *args, **kwargs: shipped_calls.append(args) or torch.zeros(8, DIM))
    profile = _calibrated_profile(PLACEHOLDER, "c" * 64)
    for artifact in (None, replace(_artifact(), method_id=PER_MESSAGE_METHOD)):
        with pytest.raises(alignment_dispatch.AlignmentRefusal):
            alignment_dispatch.aligner_for(profile, artifact)
    assert shipped_calls == []


def test_the_shipped_method_ignores_an_artifact_as_before(monkeypatch):
    torch = pytest.importorskip("torch")
    calls = []
    original = statebridge.align_hidden_sequence
    monkeypatch.setattr(statebridge, "align_hidden_sequence",
                        lambda *args, **kwargs: calls.append(args) or original(*args, **kwargs))
    aligner = alignment_dispatch.aligner_for(_message_profile(), _artifact())
    aligner(torch.randn(8, DIM), torch.randn(8, DIM), 1.0)
    assert len(calls) == 1


def test_an_unregistered_method_id_is_refused_not_dispatched():
    profile = _message_profile()
    object.__setattr__(profile, "method", "statebridge.glm53.final-prenorm-predictor.unknown.v1")
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.aligner_for(profile)
    assert caught.value.code == "profile"


def test_the_frozen_artifact_loads_only_through_the_profile_reference(artifact_file, calibrated_artifact):
    digest = calibrated_artifact.content_digest
    profile = _calibrated_profile(artifact_file, digest)
    loaded = alignment_dispatch.load_frozen_artifact(profile)
    assert statebridge_calibrated.artifact_digest(loaded) == digest
    assert loaded.binding.method_id == CALIBRATED_METHOD
    assert loaded.binding.layer_index == 77
    assert loaded.binding.model_fingerprint == FINGERPRINT
    assert alignment_dispatch.load_frozen_artifact(_message_profile()) is None
    assert alignment_dispatch.frozen_artifact_reference(_message_profile()) is None


def test_load_refuses_a_binding_for_another_model_or_layer(artifact_file, tmp_path):
    # The profile carries the runtime identity; an artifact bound for another
    # model or layer is refused at load even when its digest matches the pin.
    artifact = _artifact(seed=5)
    path = tmp_path / "other.occ3cal"
    path.write_bytes(statebridge_calibrated.serialize_artifact(artifact))
    for name, changes in (("model", {"model_fingerprint": "sha256:" + "f" * 64}),
                          ("layer", {"layer_index": 76})):
        forged = replace(artifact,
                         binding=replace(artifact.binding, **changes), content_digest=None)
        forged_bytes = statebridge_calibrated.serialize_artifact(forged)
        forged_path = tmp_path / ("forged-%s.occ3cal" % name)
        forged_path.write_bytes(forged_bytes)
        profile = _calibrated_profile(forged_path, statebridge_calibrated.artifact_digest(forged))
        with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
            alignment_dispatch.load_frozen_artifact(profile)
        assert caught.value.code == "identity"


def test_a_missing_or_swapped_file_refuses_before_loading(artifact_file, tmp_path, calibrated_artifact):
    digest = calibrated_artifact.content_digest
    absent = _calibrated_profile(tmp_path / "absent.occ3cal", digest)
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.load_frozen_artifact(absent)
    assert caught.value.code == "unavailable"
    swapped = _calibrated_profile(artifact_file, "b" * 64)
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.load_frozen_artifact(swapped)
    assert caught.value.code == "identity"
    truncated = tmp_path / "truncated.occ3cal"
    truncated.write_bytes(artifact_file.read_bytes()[:64])
    with pytest.raises(alignment_dispatch.AlignmentRefusal) as caught:
        alignment_dispatch.load_frozen_artifact(_calibrated_profile(truncated, digest))
    assert caught.value.code == "profile"


def test_a_profile_reference_is_all_or_nothing_and_message_method_takes_none(artifact_file):
    digest = "c" * 64
    for changes in ({"calibration_path": str(artifact_file)},
                    {"calibration_digest": digest},
                    {"calibration_digest": "not-a-digest"},
                    {"calibration_path": "relative/path", "calibration_digest": digest},
                    {"calibration_path": "", "calibration_digest": digest}):
        fields = _fields(method=CALIBRATED_METHOD)
        fields.update(changes)
        with pytest.raises(LatentError):
            RuntimeProfile(**fields)
    with pytest.raises(LatentError) as caught:
        RuntimeProfile(**_fields(calibration_path=str(artifact_file), calibration_digest=digest))
    assert caught.value.code == "profile"


def test_capture_metadata_reports_the_method_and_only_the_digest_it_has():
    from occ3_sglang.latent_worker import method_metadata
    message = method_metadata(_message_profile())
    assert message == {"method_id": PER_MESSAGE_METHOD}
    calibrated = method_metadata(_calibrated_profile(PLACEHOLDER, "c" * 64), "d" * 64)
    assert calibrated == {"method_id": CALIBRATED_METHOD, "artifact_digest": "d" * 64}
    # Scalars only: nothing that could carry rows or tensor content.
    assert all(type(value) is str for value in calibrated.values())


def test_the_worker_resolves_one_aligner_and_calls_it_once():
    from occ3_sglang import latent_worker
    source = Path(latent_worker.__file__).read_text(encoding="utf-8")
    assert source.count("self.aligner = aligner_for(profile, self.calibration)") == 1
    assert source.count("aligned, _ = self.aligner(h, embeddings, self.target_norm)") == 1
    assert "align_hidden_sequence(" not in source
    assert "method_metadata(self.profile, self.calibration_digest)" in source
    assert '"artifact_digest"' in source
    # The renorm step is serving-config driven: N0 skips the norm measurement.
    assert "_serving_renorm(self.aligner)" in source
    # Artifact bytes are charged at install and reported per rank and aggregate.
    assert "artifact_resident_bytes(self.calibration)" in source
    assert '"artifact_bytes_all_ranks"' in source
    signature = inspect.signature(latent_worker.WorkerRuntime.__init__)
    assert signature.parameters["artifact"].default is None
    assert "artifact" in inspect.signature(latent_worker.install_worker).parameters
