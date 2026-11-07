"""CPU diagnostics tests: these do not prove GPU/TP or semantic fidelity."""
import json
import logging
from types import SimpleNamespace as NS

import pytest
from occ3_sglang import token_trace as trace


@pytest.fixture(autouse=True)
def enabled(monkeypatch, caplog):
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    caplog.set_level(logging.INFO, logger=trace.LOG.name)


def rows(caplog):
    return [json.loads(r.message[len(trace.PREFIX):]) for r in caplog.records]


@pytest.mark.parametrize("value", [None, "0", "true", ""])
def test_off_by_default(monkeypatch, caplog, value):
    if value is None:
        monkeypatch.delenv("OCC3_DIAG_TOKEN_TRACE", raising=False)
    else:
        monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", value)
    trace.capture(None, None, None, None, None, None)
    assert not caplog.records


@pytest.mark.parametrize("a,b,index", [([1,2],[1,2],None), ([1,2],[9,2],0),
    ([1,2],[1,9],1), ([1],[1,2],1), ([1,2],[1],1), ([],[],None)])
def test_mismatch(a,b,index):
    assert trace.first_mismatch(a,b) == index


def test_bounds_and_escaping(caplog):
    tokenizer = NS(decode=lambda *a, **kw: "\n雪"*10000)
    trace.emit("test", labels=trace.summary(list(range(10000)), tokenizer))
    message = caplog.records[-1].message
    assert len(message) <= trace.MAX_LOG_CHARS
    assert "\n" not in message
    assert len(rows(caplog)[0]["labels"]["first_ids"]) == 64
    assert len(rows(caplog)[0]["labels"]["first_text"]) == 512
    trace.emit("test", oversized="x"*trace.MAX_LOG_CHARS)
    assert rows(caplog)[-1]["diag_error"] == "size_limit"


def test_isolation(monkeypatch, caplog):
    def fail(*a, **kw):
        raise RuntimeError("do-not-log-this")
    monkeypatch.setattr(trace.LOG, "info", fail)
    trace.capture_text(NS(rank=0), NS(occ3_trace_labels=(1,)), NS(tokenizer=None))
    monkeypatch.setattr(trace, "summary", fail)
    trace.capture_text(NS(rank=0), NS(occ3_trace_labels=(1,)), NS(tokenizer=None))
    assert not caplog.records


def test_capture_consume_geometry(caplog, monkeypatch):
    class Tensor:
        def __init__(self, values):
            self.values = values
            self.shape = (len(values), 2)
        def tolist(self):
            return self.values
        def numel(self):
            return len(self.values)
        def __getitem__(self, key):
            result = self.values[key]
            return Tensor(result) if isinstance(key, slice) else result
        def clone(self):
            return Tensor(self.values[:])
    torch = NS(tensor=Tensor, zeros_like=lambda x: Tensor([0]*len(x.values)))
    monkeypatch.setattr(trace, "tensor_hash", lambda t: trace.ids_hash(t.values))
    record = NS(prefix=Tensor([1,2]))
    runtime = NS(rank=0)
    trace.capture(runtime, record, NS(origin_input_ids=[5]*10, output_ids=[8,9]),
                  torch.tensor([8,7]), 0, 2)
    capture = rows(caplog)[0]
    assert capture["first_mismatch"] == 1
    assert (capture["predictor_start"],capture["predictor_end"]) == (9,11)
    desc = NS(prefix_offset=4, expected_prefix_tokens=2)
    req = NS(origin_input_ids=[3]*4+[0,0]+[6]*3, occ3_private_kv=True)
    trace.admit(runtime, record, req, NS(tokenizer=None), desc)
    assert rows(caplog)[-1]["holes"]["first_ids"] == [0,0]
    item = NS(prefix_len=3, extend_len=4, descriptor=desc)
    batch = NS(positions=torch.tensor([3,4,6,6]), out_cache_loc=torch.tensor([20,21,22,23]))
    trace.chunk(runtime, record, item, batch, 1, 2, 0, "target")
    chunk = rows(caplog)[-1]
    assert chunk["first_position_mismatch"] == 1
    assert chunk["kv_slots"]["first_ids"] == [21,22]
    trace.copied(runtime, record, record.prefix.clone(), 0, 2, 0, "target")
    assert rows(caplog)[-1]["copy_matches"] is True
    record.prefix.values[0] += 1
    trace.copied(runtime, record, torch.zeros_like(record.prefix), 0, 2, 0, "target")
    assert rows(caplog)[-1]["prefix_matches_capture"] is False
    assert rows(caplog)[-1]["copy_matches"] is False


def test_rank_and_decoder_failure(caplog):
    trace.capture_text(NS(rank=1), NS(occ3_trace_labels=(1,)), NS())
    class Broken:
        def decode(self, *a, **kw):
            raise ValueError("private")
    trace.capture_text(NS(rank=0), NS(occ3_trace_labels=(1,)), NS(tokenizer=Broken()))
    assert not caplog.records


def test_attested():
    from occ3_sglang.native_install import _REQUIRED_PLUGIN
    assert "token_trace.py" in _REQUIRED_PLUGIN


def test_tensor_hash_cpu():
    torch = pytest.importorskip("torch")
    tensor = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.bfloat16)
    assert trace.tensor_hash(tensor) == trace.tensor_hash(tensor.clone())
    assert trace.tensor_hash(tensor) != trace.tensor_hash(tensor + 1)
    assert trace.tensor_hash(torch.zeros((129,2))) is None


@pytest.mark.parametrize("slots,mismatch", [([21,22], None), ([21,99], 1)])
def test_terminal_mapping(caplog, slots, mismatch):
    class Pool:
        def __getitem__(self, key):
            assert key == (7, slice(4,6))
            return NS(tolist=lambda: slots)
    record = NS(occ3_trace_slots={4:21, 5:22}, occ3_trace_hash="a"*64)
    trace.settled(NS(rank=0), record, NS(req_to_token_pool=NS(req_to_token=Pool())),
                  NS(req_pool_idx=7, retraction_count=0),
                  NS(prefix_offset=4, expected_prefix_tokens=2))
    assert rows(caplog)[-1]["initial_mapping_complete"] is True
    assert rows(caplog)[-1]["first_slot_mismatch"] == mismatch


def test_no_transport_fields(caplog):
    record = NS(occ3_trace_labels=(1,2), handle="forbidden-handle",
                principal_digest="forbidden-principal", receipt=b"forbidden-receipt")
    trace.capture_text(NS(rank=0), record, NS(tokenizer=None))
    assert "forbidden" not in caplog.text


def test_zero_overlap_chunk(caplog):
    class Vector:
        def __init__(self, values):
            self.values = values
        def __getitem__(self, key):
            return Vector(self.values[key]) if isinstance(key, slice) else self.values[key]
        def tolist(self):
            return self.values
    batch = NS(positions=Vector([0,1]), out_cache_loc=Vector([10,11]))
    item = NS(prefix_len=0, extend_len=2, descriptor=NS(prefix_offset=4, expected_prefix_tokens=2))
    trace.chunk(NS(rank=0), NS(), item, batch, 2, 0, 0, "target")
    row = rows(caplog)[-1]
    assert row["rows"] == 0
    assert row["chunk_end"] == 2
    assert row["first_position_mismatch"] is None
