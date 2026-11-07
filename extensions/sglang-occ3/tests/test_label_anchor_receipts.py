"""Backward-compatible, bounded source receipt metadata (no GPU)."""
import json
import pytest
from occ3_sglang.attachment_contract import AttachmentError, SourceCaptureReceipt
from occ3_sglang.use_receipts import source_receipt_upper_bound, source_receipt_v2_upper_bound


@pytest.mark.parametrize("factory", [source_receipt_upper_bound, source_receipt_v2_upper_bound])
def test_label_anchor_receipt_roundtrip_and_legacy_default(factory):
    wire = json.loads(factory(limit=4096))
    wire.pop("label_anchor_alpha", None)
    assert SourceCaptureReceipt.parse(wire).label_anchor_alpha == 0.0
    assert SourceCaptureReceipt.parse(wire).wire() == wire
    assert SourceCaptureReceipt.parse(dict(wire, label_anchor_alpha=0.0)).wire() == wire
    for alpha in (0.0, 0.3, 0.5, 0.7, 1.0):
        wire["label_anchor_alpha"] = alpha
        assert SourceCaptureReceipt.parse(wire).wire().get("label_anchor_alpha", 0.0) == alpha


@pytest.mark.parametrize("alpha", [True, False, None, "0.3", -0.3, 0.1, 1.1, float("nan"), float("inf")])
def test_label_anchor_receipt_rejects_invalid_values(alpha):
    wire = json.loads(source_receipt_upper_bound())
    wire["label_anchor_alpha"] = alpha
    with pytest.raises(AttachmentError):
        SourceCaptureReceipt.parse(wire)
