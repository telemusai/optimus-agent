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


def test_kv_values(monkeypatch, caplog):
    buf = NS(shape=(10, 1, 2), element_size=lambda: 1)
    pool = NS(size=10, kv_buffer=[buf])
    data = {2: b"ab", 4: b"cd"}
    calls = []
    def copy(kind, buffer, slots):
        calls.append(slots)
        return b"".join(data[i] for i in slots)
    monkeypatch.setattr(trace, "kv_bytes", copy)
    record = NS()
    for point in ("prefill_complete", "first_decode"):
        trace.kv_snapshot(NS(rank=0), record, pool, [2, 4], point)
    assert rows(caplog)[-1]["changed"] is False
    assert rows(caplog)[-1]["sha256"] == rows(caplog)[-2]["sha256"]
    data[4] = b"ce"
    trace.kv_snapshot(NS(rank=0), record, pool, [2, 4], "terminal_settle")
    assert rows(caplog)[-1]["changed"] is True
    assert rows(caplog)[-1]["first_changed_slot"] == 4
    assert rows(caplog)[-1]["bytes_copied"] == 4
    for invalid in ([-1], [10], [None]):
        trace.kv_snapshot(NS(rank=0), record, pool, invalid, "invalid")
        assert rows(caplog)[-1]["skipped"] == "invalid_slots"
    assert len(calls) == 3
    monkeypatch.delenv("OCC3_DIAG_TOKEN_TRACE")
    trace.kv_snapshot(None, None, None, None, None)
    trace.kv_forward(None, None, None)
    assert len(calls) == 3


def test_kv_rank_and_buffer_bounds(monkeypatch, caplog):
    monkeypatch.setattr(trace, "kv_bytes", lambda *a: pytest.fail("unexpected copy"))
    trace.kv_snapshot(NS(rank=1), None, None, None, "prefill_complete")
    buf = NS(shape=(2, 1, 2), element_size=lambda: 1)
    trace.kv_snapshot(NS(rank=0), NS(), NS(size=10, kv_buffer=[buf]), [3], "test")
    assert rows(caplog)[-1]["skipped"] == "invalid_buffer_slots"


def test_kv_sampling_and_index_layout(monkeypatch, caplog):
    mla = NS(shape=(100, 1, 2), element_size=lambda: 1)
    index = NS(shape=(2, 64 * 132), element_size=lambda: 1)
    pool = NS(size=100, kv_buffer=[mla], index_k_with_scale_buffer=[index],
              page_size=64, index_head_dim=128, index_kpool=1, index_kpool_compress=False)
    monkeypatch.setattr(trace, "KV_COPY_LIMIT", 268)
    copied = []
    def copy(kind, buf, slots):
        copied.append((kind, slots))
        return b"a" * len(slots) * (2 if kind == "mla" else 132)
    monkeypatch.setattr(trace, "kv_bytes", copy)
    trace.kv_snapshot(NS(rank=0), NS(), pool, [1, 2, 3, 4], "prefill_complete")
    assert copied == [("mla", [1, 4]), ("index", [1, 4])]
    assert rows(caplog)[-1]["sampled"] is True
    assert rows(caplog)[-1]["bytes_copied"] == 268
    pool.index_kpool = 2
    trace.kv_snapshot(NS(rank=0), NS(), pool, [1], "test")
    assert rows(caplog)[-1]["skipped"] == "unsupported_index_layout"


def test_kv_forward_points(monkeypatch):
    from occ3_sglang import latent_worker as worker
    descriptor = object.__new__(worker.ConsumeWorkerDescriptor)
    object.__setattr__(descriptor, "prefix_offset", 4)
    object.__setattr__(descriptor, "expected_prefix_tokens", 2)
    record = NS(occ3_trace_slots={4: 2})
    runtime = NS(rank=0, resolve=lambda d: record)
    monkeypatch.setattr(worker, "_runtime", lambda: runtime)
    points = []
    monkeypatch.setattr(trace, "kv_snapshot", lambda *a: points.append(a[-1]))
    runner = NS(token_to_kv_pool=object())
    mode = NS(is_extend=lambda: True, is_target_verify=lambda: False)
    batch = NS(occ3_context=[NS(descriptor=descriptor)], forward_mode=mode)
    trace.kv_forward(runner, batch, "prefill_complete")
    assert not points
    record.occ3_trace_slots[5] = 3
    trace.kv_forward(runner, batch, "prefill_complete")
    trace.kv_forward(runner, batch, "prefill_complete")
    trace.kv_forward(runner, batch, "first_decode")
    assert points == ["prefill_complete"]
    mode.is_target_verify = lambda: True
    trace.kv_forward(runner, batch, "first_decode")
    trace.kv_forward(runner, batch, "first_decode")
    assert points == ["prefill_complete", "first_decode"]
