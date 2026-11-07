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


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_copied_per_rank_and_missing_capture(monkeypatch, caplog, rank):
    class Matrix:
        def __init__(self, values): self.values = values
        def __getitem__(self, key): return Matrix(self.values[key])
    monkeypatch.setattr(trace, "tensor_hash", lambda value: trace.ids_hash(value.values))
    record = NS(prefix=Matrix([1, 2]))
    if rank == 0:
        record.occ3_trace_prefix_hash = trace.ids_hash([1, 2])
    trace.copied(NS(rank=rank), record, Matrix([1, 2]), 0, 2, 0, "target")
    row = rows(caplog)[-1]
    assert row["rank"] == rank and row["copy_matches"] is True
    assert row["prefix_matches_capture"] is (True if rank == 0 else None)
    trace.copied(NS(rank=rank), record, Matrix([1, 9]), 0, 2, 0, "draft")
    assert rows(caplog)[-1]["copy_matches"] is False
    monkeypatch.delenv("OCC3_DIAG_TOKEN_TRACE")
    monkeypatch.setattr(trace, "tensor_hash", lambda _: pytest.fail("disabled hash"))
    trace.copied(None, None, None, None, None, None, None)


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_saved_fence_failure_is_visible_and_isolated(monkeypatch, caplog, rank):
    events = []
    def fail_fence():
        events.append("fence")
        raise RuntimeError("private-exception-text")
    monkeypatch.setattr(trace, "kv_snapshot", lambda *args: pytest.fail("unfenced read"))
    runtime = NS(rank=rank, fence=fail_fence)
    record = NS(occ3_trace_kv_pool=object(), occ3_trace_slots={4:2, 5:3})
    descriptor = NS(prefix_offset=4, expected_prefix_tokens=2)
    trace.kv_saved(runtime, record, descriptor, "consume_released", fence=True)
    assert rows(caplog)[-1] == {"event": "consume_kv_values", "rank": rank,
                               "point": "consume_released", "skipped": "fence_failed"}
    assert "private-exception-text" not in caplog.text
    assert events == ["fence"]
    monkeypatch.delenv("OCC3_DIAG_TOKEN_TRACE")
    trace.kv_saved(runtime, record, descriptor, "consume_released", fence=True)
    assert events == ["fence"]


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


def test_terminal_saved_mapping_without_live_pool(monkeypatch, caplog):
    rank = 0
    class FreedScheduler:
        @property
        def req_to_token_pool(self):
            pytest.fail("terminal settle must not read a freed pool row")
    monkeypatch.setattr(trace, "kv_snapshot", lambda *a: pytest.fail("late KV read"))
    monkeypatch.setattr(trace, "tensor_hash", lambda *a: pytest.fail("late tensor hash"))
    record = NS(occ3_trace_slots={4:21, 5:22}, occ3_trace_hash="a"*64)
    trace.settled(NS(rank=rank), record, FreedScheduler(),
                  NS(kv=NS(req_pool_idx=None), retraction_count=3),
                  NS(prefix_offset=4, expected_prefix_tokens=2))
    row = rows(caplog)[-1]
    assert row["kv_slots"]["first_ids"] == [21, 22]
    assert row["initial_mapping_complete"] is True
    assert row["mapping_source"] == "injection"
    assert row["num_retractions"] == 3
    assert "first_slot_mismatch" not in row


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
    def copy(buffers, slots):
        calls.append(slots)
        return [b"".join(data[i] for i in slots)]
    monkeypatch.setattr(trace, "kv_point_bytes", copy)
    record = NS()
    for point in ("prefill_complete", "consume_released"):
        trace.kv_snapshot(NS(rank=0), record, pool, [2, 4], point)
    assert rows(caplog)[-1]["changed"] is False
    assert rows(caplog)[-1]["sha256"] == rows(caplog)[-2]["sha256"]
    data[4] = b"ce"
    trace.kv_snapshot(NS(rank=0), record, pool, [2, 4], "private_release")
    assert rows(caplog)[-1]["changed"] is True
    assert rows(caplog)[-1]["first_changed_slot"] == 4
    assert rows(caplog)[-1]["bytes_copied"] == 4
    for invalid, reason in (([-1], "invalid_slots"), ([10], "invalid_buffer_slots"),
                            ([None], "invalid_slots")):
        trace.kv_snapshot(NS(rank=0), record, pool, invalid, "invalid")
        assert rows(caplog)[-1]["skipped"] == reason
    assert len(calls) == 3
    monkeypatch.delenv("OCC3_DIAG_TOKEN_TRACE")
    trace.kv_snapshot(None, None, None, None, None)
    trace.kv_forward(None, None, None)
    assert len(calls) == 3


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_kv_rank_and_buffer_bounds(monkeypatch, caplog, rank):
    monkeypatch.setattr(trace, "kv_point_bytes", lambda *a: pytest.fail("unexpected copy"))
    buf = NS(shape=(2, 1, 2), element_size=lambda: 1)
    trace.kv_snapshot(NS(rank=rank), NS(), NS(size=10, kv_buffer=[buf]), [3], "test")
    assert rows(caplog)[-1]["skipped"] == "invalid_buffer_slots"
    assert rows(caplog)[-1]["rank"] == rank


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_kv_final_allocated_page_and_physical_bounds(monkeypatch, caplog, rank):
    # Allocator pages are 1..size//page_size; page zero is padding. Both
    # real buffer layouts therefore include the final page above logical size.
    mla = NS(shape=(192, 1, 2), element_size=lambda: 1)
    index = NS(shape=(3, 64 * 132), element_size=lambda: 1)
    pool = NS(size=128, page_size=64, kv_buffer=[mla],
              index_k_with_scale_buffer=[index], index_head_dim=128,
              index_kpool=1, index_kpool_compress=False)
    copies = []
    def copy(buffers, slots):
        copies.append(list(slots))
        return [b"a" * len(slots) * width for _, _, _, width in buffers]
    monkeypatch.setattr(trace, "kv_point_bytes", copy)
    record = NS()
    for point in ("prefill_complete", "consume_released", "private_release"):
        trace.kv_snapshot(NS(rank=rank), record, pool, [128, 191], point)
        row = rows(caplog)[-1]
        assert row["rank"] == rank and row["point"] == point
        assert row["hashed_slots"] == 2 and "skipped" not in row
    assert rows(caplog)[-1]["changed"] is False
    # First slot past storage is reported, not silently dropped or gathered.
    trace.kv_snapshot(NS(rank=rank), record, pool, [192], "private_release")
    row = rows(caplog)[-1]
    assert row["skipped"] == "invalid_buffer_slots"
    assert row["point"] == "private_release" and row["rank"] == rank
    assert "changed" not in row
    # The narrower index allocation also constrains reads independently of MLA.
    index.shape = (2, 64 * 132)
    trace.kv_snapshot(NS(rank=rank), record, pool, [128], "private_release")
    assert rows(caplog)[-1]["skipped"] == "invalid_buffer_slots"
    assert copies == [[128, 191]] * 3


@pytest.mark.parametrize("limit", [1, 15, 16, 20, 32])
def test_kv_sampling_preserves_tail(monkeypatch, caplog, limit):
    mla = NS(shape=(100, 1, 2), element_size=lambda: 1)
    index = NS(shape=(2, 64 * 132), element_size=lambda: 1)
    pool = NS(size=100, kv_buffer=[mla], index_k_with_scale_buffer=[index],
              page_size=64, index_head_dim=128, index_kpool=1, index_kpool_compress=False)
    monkeypatch.setattr(trace, "KV_COPY_LIMIT", 134 * limit)
    copied = []
    def copy(buffers, slots):
        copied.append(list(slots))
        return [b"a" * len(slots) * width for _, _, _, width in buffers]
    monkeypatch.setattr(trace, "kv_point_bytes", copy)
    slots = list(range(32))
    trace.kv_snapshot(NS(rank=1), NS(), pool, slots, "prefill_complete")
    row = rows(caplog)[-1]
    if limit < 16:
        assert row["skipped"] == "layout_exceeds_budget"
        assert not copied
    else:
        assert len(copied) == 1
        assert len(copied[0]) <= limit
        assert len(set(copied[0])) == len(copied[0])
        assert set(slots[-16:]) <= set(copied[0])
        assert row["bytes_copied"] <= 134 * limit
        assert row["sampled"] is (limit < 32)
        assert row["blocking_copies"] == 1
    pool.index_kpool = 2
    trace.kv_snapshot(NS(rank=1), NS(), pool, [1], "test")
    assert rows(caplog)[-1]["skipped"] == "unsupported_index_layout"


def test_kv_short_prefix_must_fit_entire_tail(monkeypatch, caplog):
    buf = NS(shape=(10, 1, 2), element_size=lambda: 1)
    monkeypatch.setattr(trace, "KV_COPY_LIMIT", 4)
    monkeypatch.setattr(trace, "kv_point_bytes", lambda *a: pytest.fail("unexpected copy"))
    trace.kv_snapshot(NS(rank=0), NS(), NS(size=10, kv_buffer=[buf]), [1, 2, 3], "prefill_complete")
    assert rows(caplog)[-1]["skipped"] == "layout_exceeds_budget"


def descriptor_for(worker):
    from dataclasses import fields
    descriptor = object.__new__(worker.ConsumeWorkerDescriptor)
    for field in fields(worker.ConsumeWorkerDescriptor):
        object.__setattr__(descriptor, field.name, None)
    for key, value in dict(prefix_offset=4, expected_prefix_tokens=2,
                           attempt_id="attempt", native_rid="rid", reservation_id="reservation").items():
        object.__setattr__(descriptor, key, value)
    return descriptor


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_kv_forward_points(monkeypatch, rank):
    from occ3_sglang import latent_worker as worker
    descriptor = descriptor_for(worker)
    record = NS(occ3_trace_slots={4: 2})
    runtime = NS(rank=rank, resolve=lambda d: record)
    monkeypatch.setattr(worker, "_runtime", lambda: runtime)
    points = []
    monkeypatch.setattr(trace, "kv_snapshot", lambda *a: points.append(a[-1]))
    runner = NS(token_to_kv_pool=object())
    batch = NS(occ3_context=[NS(descriptor=descriptor)])
    trace.kv_forward(runner, batch, "prefill_complete")
    assert not points
    record.occ3_trace_slots[5] = 3
    trace.kv_forward(runner, batch, "prefill_complete")
    trace.kv_forward(runner, batch, "prefill_complete")
    trace.kv_forward(runner, batch, "first_decode")
    # Real shared decode batches cannot carry the eager consume context.
    trace.kv_forward(runner, NS(occ3_context=None), "prefill_complete")
    trace.kv_forward(runner, NS(occ3_context=None), "first_decode")
    assert points == ["prefill_complete"]


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_release_hooks_snapshot_before_eager_clear_and_pool_free(monkeypatch, rank):
    from occ3_sglang import latent_worker as worker
    descriptor = descriptor_for(worker)
    record = NS(consumer=descriptor, state="injected", injected_positions={4, 5},
                occ3_trace_slots={4:2, 5:3}, occ3_trace_kv_pool=object(), inflight=1)
    req = NS(occ3_worker_descriptor=descriptor, occ3_force_eager=True,
             rid="rid", kv=NS(req_pool_idx=7))
    events = []
    runtime = NS(rank=rank, registry=NS(source_for_attempt=lambda _: record),
                 fence=lambda: events.append("fence"),
                 preflight=lambda name, operation: operation())
    monkeypatch.setattr(worker, "_runtime", lambda *a: runtime)
    monkeypatch.setattr(worker, "_cleanup_runtime", lambda *a: runtime)
    def snapshot(rt, rec, pool, slots, point):
        assert rt is runtime and rec is record and pool is record.occ3_trace_kv_pool
        assert slots == [2, 3]
        assert req.kv.req_pool_idx == 7
        if point == "consume_released":
            assert req.occ3_force_eager is True
        events.append(point)
    monkeypatch.setattr(trace, "kv_snapshot", snapshot)
    assert worker.release_consume_cohort(NS(), req) is True
    assert req.occ3_force_eager is False
    assert events == ["fence", "consume_released"]
    worker.fence_private_kv_release(NS(), req)
    assert events == ["fence", "consume_released", "fence", "private_release"]
    assert record.inflight == 0


@pytest.mark.parametrize("rank", [0, 1, 7])
def test_chunk_records_slots_on_all_ranks(caplog, rank):
    class Vector:
        ndim = 1
        def __init__(self, values): self.values = values
        def numel(self): return len(self.values)
        def __getitem__(self, key):
            return Vector(self.values[key]) if isinstance(key, slice) else self.values[key]
        def tolist(self): return self.values
    record = NS()
    item = NS(prefix_len=4, extend_len=2, descriptor=NS(prefix_offset=4, expected_prefix_tokens=2))
    batch = NS(positions=Vector([4, 5]), out_cache_loc=Vector([21, 22]),
               occ3_dsa_pin=tuple(Vector(v) for v in ([4], [2], [1])))
    trace.chunk(NS(rank=rank), record, item, batch, 0, 2, 0, "target")
    assert record.occ3_trace_slots == {4:21, 5:22}
    row = rows(caplog)[-1]
    assert row["rank"] == rank
    assert row["dsa_pin"] == [[4], [2], [1]]


def test_real_mla_and_paged_index_bytes(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.version, "hip", None)
    mla = torch.arange(70 * 3, dtype=torch.int64).to(torch.uint8).reshape(70, 1, 3)
    index = torch.zeros((2, 64 * 132), dtype=torch.uint8)
    slots = [63, 64, 1]
    expected = []
    for i, slot in enumerate(slots):
        page, row = divmod(slot, 64)
        index[page, row*128:(row+1)*128] = 11 + i
        index[page, 8192+row*4:8192+(row+1)*4] = 91 + i
        expected.append(bytes([11+i])*128 + bytes([91+i])*4)
    expected_mla = bytes(value for slot in slots for value in mla[slot].flatten().tolist())
    assert trace.kv_bytes("mla", mla, slots) == expected_mla
    assert trace.kv_bytes("index", index, slots) == b"".join(expected)
    buffers = [("mla", 0, mla, 3), ("index", 0, index, 132)]
    transfers = []
    original_cpu = torch.Tensor.cpu
    def counted_cpu(tensor, *args, **kwargs):
        transfers.append(tensor.numel())
        return original_cpu(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "cpu", counted_cpu)
    assert trace.kv_point_bytes(buffers, slots) == [expected_mla, b"".join(expected)]
    assert transfers == [len(slots) * 135]


def test_alignment_cosines_bounded_and_rounded(caplog):
    torch = pytest.importorskip("torch")
    prefix = torch.tensor([[1., 0.], [1., 0.], [0., 0.], [1., 1.]])
    embeddings = torch.tensor([[1., 0.], [-1., 0.], [1., 0.], [1., 0.]])
    trace.alignment(NS(rank=0), prefix, embeddings)
    row = rows(caplog)[-1]
    assert row["event"] == "alignment"
    assert row["row_cosine"] == [1.0, -1.0, 0.0, 0.707107]
    assert row["rows"] == 4 and row["rank"] == 0
    caplog.clear()
    trace.alignment(NS(rank=0), torch.ones((128, 2)), torch.ones((128, 2)))
    assert len(rows(caplog)[-1]["row_cosine"]) == 128
    caplog.clear()
    trace.alignment(NS(rank=0), torch.ones((129, 2)), torch.ones((129, 2)))
    assert not caplog.records


def test_alignment_neighbors_and_stages(caplog):
    torch = pytest.importorskip("torch")
    e = torch.eye(3)
    runtime = NS(rank=0, embedding=NS(num_embeddings=3),
                 profile=NS(anchor_alpha=0.3), target_norm=2.0)
    tokens = torch.tensor([0, 1, 2])
    for stage in ("raw", "aligned", "final"):
        trace.alignment_stage(runtime, e[[1, 2, 0]], e, tokens, stage)
    result = rows(caplog)
    assert len(result) == 9
    assert result[0]["cosine_offsets"] == [None, 0.0, 1.0]
    assert result[1]["cosine_offsets"] == [0.0, 0.0, 1.0]
    assert result[2]["cosine_offsets"] == [0.0, 0.0, None]
    assert result[0]["row_norm"] == result[0]["embedding_norm"] == 1.0
    assert result[0]["target_norm"] == 2.0
    assert result[0]["anchor_alpha"] == 0.3
    trace.alignment_stage(runtime, e, e, torch.tensor([-1, 1, 2]), "raw")
    assert len(rows(caplog)) == 9


def test_nearest_shard_padding_and_ties(caplog):
    torch = pytest.importorskip("torch")
    embedding = NS(weight=torch.tensor([[1.,0.], [0.,1.], [9.,9.], [1.,0.]]),
        embedding_dim=2, num_embeddings=8, shard_indices=NS(
        org_vocab_start_index=2, org_vocab_end_index=4,
        added_vocab_start_index=7, added_vocab_end_index=8, num_org_elements_padded=3))
    trace.alignment_nearest(NS(rank=1, embedding=embedding), torch.eye(2), torch.tensor([2, 3]))
    result = rows(caplog)
    assert [r["top1_id"] for r in result] == [2, 3]
    assert all(r["top1_hit"] and r["top1_cosine"] == 1.0 for r in result)
    assert all(r["rank"] == 1 for r in result)


def test_alignment_disabled_no_access(monkeypatch, caplog):
    monkeypatch.delenv("OCC3_DIAG_TOKEN_TRACE")
    trace.alignment_stage(None, None, None, None, "raw")
    trace.alignment_nearest(None, None, None)
    assert not rows(caplog)


def test_alignment_cast_precision(caplog):
    torch = pytest.importorskip("torch")
    solved = torch.tensor([[1.0003, 0.5001], [0.0, 0.0]])
    copied = solved.to(torch.float16)
    original = solved.clone()
    trace.alignment_cast(NS(rank=0, profile=NS(method="test", anchor_alpha=0.3)), copied, solved, copied)
    result = rows(caplog)
    assert result[0]["solver_dtype"] == "torch.float32"
    assert result[0]["buffer_dtype"] == "torch.float16"
    assert result[0]["cosine"] == 1.0
    assert result[0]["max_abs_diff"] == 0.0003
    assert result[1]["cosine"] is None
    assert torch.equal(original, solved)


def test_alignment_fp8_baseline(caplog):
    torch = pytest.importorskip("torch")
    values = torch.tensor([[1.1, 0.7], [0.0, 0.0]])
    trace.alignment_precision(NS(rank=0), values, values)
    result = rows(caplog)
    assert len(result) == 2
    assert result[0]["fp8_cosine"] == result[0]["embedding_fp8_cosine"]
    if result[0]["fp8_skipped"] is None:
        expected = trace.precision_cosine(values, values.to(torch.float8_e4m3fn).float())
        assert result[0]["fp8_cosine"] == expected[0]
        assert expected[0] >= 0.999
        assert result[1]["fp8_cosine"] is None
    assert result[0]["fp32_end_to_end_skipped"] == "requires_solver_and_global_anchor_replay"


def test_precision_off_and_isolated(monkeypatch, caplog):
    monkeypatch.delenv("OCC3_DIAG_TOKEN_TRACE")
    trace.alignment_cast(None, None, None, None)
    trace.alignment_precision(None, None, None)
    monkeypatch.setenv("OCC3_DIAG_TOKEN_TRACE", "1")
    trace.alignment_cast(NS(rank=0), None, None, None)
    trace.alignment_precision(NS(rank=0), None, None)
    assert not rows(caplog)


def test_logit_lens_offsets_shard_and_one_copy(caplog, monkeypatch):
    torch = pytest.importorskip("torch")
    norm = type("RMSNorm", (), {})()
    norm.weight = torch.ones(3)
    norm.variance_epsilon = 1e-6
    norm.variance_size_override = norm.override_orig_dtype = None
    norm.cast_x_before_out_mul = False
    head = NS(weight=torch.eye(3), bias=None, embedding_dim=3, num_embeddings=3,
        quant_method=type("UnquantizedEmbeddingMethod", (), {})(),
        shard_indices=NS(org_vocab_start_index=0, org_vocab_end_index=3,
                         added_vocab_start_index=3, added_vocab_end_index=3,
                         num_org_elements_padded=3))
    runtime = NS(rank=0, profile=NS(method="probe", anchor_alpha=0.0), runner=NS(model=NS(
        model=NS(norm=norm), lm_head=head,
        logits_processor=NS(logit_scale=None, final_logit_softcapping=None))))
    count = []
    original = trace.point_copy
    def copy(*args):
        count.append(1)
        return original(*args)
    monkeypatch.setattr(trace, "point_copy", copy)
    trace.alignment_logit_lens(runtime, torch.eye(3), torch.arange(3))
    result = rows(caplog)
    assert count == [1]
    assert len(result) == 3
    assert [r["top_ids"][0] for r in result] == [0, 1, 2]
    assert result[0]["label_logits_offsets"] == [None, 1.732, 0.0]
    assert result[1]["label_logits_offsets"] == [0.0, 1.732, 0.0]
    assert result[2]["label_logits_offsets"] == [0.0, 1.732, None]
    assert result[0]["method_id"] == "probe"
    assert result[0]["vocab_start"] == 0
    head.quant_method = object()
    trace.alignment_logit_lens(runtime, torch.eye(3), torch.arange(3))
    assert rows(caplog)[-1]["reason"] == "unsupported_norm_or_head"
    monkeypatch.delenv("OCC3_DIAG_TOKEN_TRACE")
    trace.alignment_logit_lens(None, None, None)
    assert count == [1]


def test_stage_single_copy(caplog, monkeypatch):
    torch = pytest.importorskip("torch")
    original = trace.point_copy
    calls = []
    def copy(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(trace, "point_copy", copy)
    runtime = NS(rank=0, embedding=NS(num_embeddings=3),
        profile=NS(anchor_alpha=0.3, method="probe"), target_norm=torch.tensor(2.0))
    trace.alignment_stage(runtime, torch.eye(3), torch.eye(3), torch.arange(3), "raw")
    assert calls == [1]
    assert rows(caplog)[0]["target_norm"] == 2.0


def test_documented_global_top8_rank_bound():
    """A label present in a local top-8 need not have a known global rank."""
    from pathlib import Path
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert "truncate\nthe merged list to the global top-8" in readme
    assert "otherwise report `>8`" in readme
    shard_a = [(float(8-i), i) for i in range(8)]
    shard_b_full = [(float(200-i), 100+i) for i in range(100)]
    union = sorted(shard_a + shard_b_full[:8], reverse=True)
    label = 7  # Local rank 8; union rank 16, but true global rank 108.
    assert next(i+1 for i, (_, token) in enumerate(union) if token == label) == 16
    full = sorted(shard_a + shard_b_full, reverse=True)
    assert next(i+1 for i, (_, token) in enumerate(full) if token == label) == 108
    merged_top8 = union[:8]
    rank = next((i+1 for i, (_, token) in enumerate(merged_top8) if token == label), ">8")
    assert rank == ">8"
    assert merged_top8 == full[:8]


def test_documented_lens_residual_rounding_and_sync_scope():
    from pathlib import Path
    readme = (Path(__file__).parents[1] / "README.md").read_text()
    assert "rounded fused add+norm input (the post-add residual)" in readme
    assert "not a bit-exact replay" in readme
    assert "One bounded synchronous packed copy per point inside align()" in readme
    assert "there is no asynchronous staging" in readme
