"""DSA indexer pin producer (occ3_sglang.dsa_pin); no CUDA graph carries the pin."""
from types import SimpleNamespace as NS

import pytest

from occ3_sglang import dsa_pin
from test_latent_registry import consume
from test_latent_types import capture


TOPK = 2048
MARGIN = 2


@pytest.fixture(autouse=True)
def armed():
    """The latent runtime arms the threshold at install; tests do it here."""
    saved = dsa_pin._threshold
    dsa_pin.configure(TOPK, MARGIN)
    yield
    dsa_pin._threshold = saved


def attached(start=0, count=8, prompt=3000, output=0):
    return NS(occ3_attached=True, origin_input_ids=[1] * prompt, output_ids=[2] * output,
              occ3_worker_descriptor=consume(prefix_offset=start, expected_prefix_tokens=count))


def ordinary():
    return NS(occ3_attached=False, occ3_worker_descriptor=None)


def test_pin_rows_keyed_by_batch_index_only_attached_consume_rows():
    assert dsa_pin.pin_rows([ordinary(), ordinary()]) is None
    assert dsa_pin.pin_rows([]) is None
    rows = dsa_pin.pin_rows([ordinary(), attached(0, 8), ordinary()])
    assert rows == ([0, 0, 0], [0, 8, 0])
    rows = dsa_pin.pin_rows([ordinary(), ordinary(), attached(900, 96)])
    assert rows == ([0, 0, 900], [0, 0, 96])
    # A descriptor that is not attached, or a capture descriptor, never pins.
    stale = NS(occ3_attached=False, occ3_worker_descriptor=consume())
    cap = NS(occ3_attached=True, occ3_worker_descriptor=capture())
    assert dsa_pin.pin_rows([stale, cap]) is None


def test_pin_rows_rebuilt_per_batch_after_filter_and_merge():
    a = attached(5, 16)
    before = dsa_pin.pin_rows([ordinary(), a])
    assert before == ([0, 5], [0, 16])
    # filter_batch drops row 0: the pin follows the request to index 0.
    assert dsa_pin.pin_rows([a]) == ([5], [16])
    # merge_batch prepends two ordinary rows: index 2; no stale slot remains.
    assert dsa_pin.pin_rows([ordinary(), ordinary(), a]) == ([0, 0, 5], [0, 0, 16])
    # The attached request leaves: the next batch carries no pin at all.
    assert dsa_pin.pin_rows([ordinary(), ordinary()]) is None


def test_pin_rows_bounds_fail_closed():
    bad = NS(occ3_attached=True,
             occ3_worker_descriptor=NS(prefix_offset=0, expected_prefix_tokens=129))
    # A non-descriptor object is ignored, never trusted.
    assert dsa_pin.pin_rows([bad]) is None
    assert dsa_pin.PIN_LANES == 128


def test_out_of_bounds_descriptor_never_raises_in_init_new(caplog):
    """N8: pin_rows runs in ForwardBatch.init_new; it logs once, never raises."""
    import logging

    good = attached(0, 8)
    bad_values = ((0, 129), (0, 0), (2**20 + 1, 8), (-1, 8))
    rows = []
    for start, count in bad_values:
        d = consume(prefix_offset=0, expected_prefix_tokens=8)
        object.__setattr__(d, "prefix_offset", start)
        object.__setattr__(d, "expected_prefix_tokens", count)
        rows.append(NS(occ3_attached=True, occ3_worker_descriptor=d))
    dsa_pin._bounds_logged = False
    with caplog.at_level(logging.ERROR, logger=dsa_pin.__name__):
        assert dsa_pin.pin_rows(rows) is None
        assert dsa_pin.pin_rows([rows[0], good]) == ([0, 0], [0, 8])
    errors = [r for r in caplog.records if "out of bounds" in r.getMessage()]
    assert len(errors) == 1


def test_draft_pin_is_in_the_shifted_draft_row_space():
    """Fable D#1: draft input is shifted by one (prefix_span shift=1)."""
    torch = pytest.importorskip("torch")
    from occ3_sglang.prefix_layout import prefix_span

    offset, count = 40, 8
    a = attached(offset, count)
    batch = NS(reqs=[ordinary(), a])
    target = dsa_pin.build(batch, "cpu", record=True)
    draft = dsa_pin.build(batch, "cpu", record=False)
    assert target[0].tolist() == [0, offset] and draft[0].tolist() == [0, offset - 1]
    assert target[1].tolist() == draft[1].tolist() == [0, count]
    # The draft pin covers exactly the draft rows that receive prefix rows.
    local, n, source = prefix_span(count, offset, 0, 100, shift=1)
    assert source == 0 and n == count
    pinned = set(range(int(draft[0][1]), int(draft[0][1]) + count))
    assert set(range(local, local + n)) == pinned
    # Offset 0 (prepend): prefix row 0 never reaches the draft; start stays 0.
    p = attached(0, count)
    assert dsa_pin.build(NS(reqs=[p]), "cpu", record=False)[0].tolist() == [0]
    assert dsa_pin.pin_rows([p], shift=1) == ([0], [count])
    # Explicit draft flag wins over the record default.
    assert dsa_pin.build(batch, "cpu", record=False, draft=False)[0].tolist() == [0, offset]


def test_build_records_forced_on_target_only():
    torch = pytest.importorskip("torch")
    a = attached(3, 8)
    batch = NS(reqs=[ordinary(), a, ordinary()])
    start, count, forced = dsa_pin.build(batch, "cpu", record=False)
    assert start.dtype == count.dtype == forced.dtype == torch.int32
    # record=False is the draft runner: shifted draft row space (start - 1).
    assert start.tolist() == [0, 2, 0] and count.tolist() == [0, 8, 0]
    assert forced.tolist() == [0, 0, 0]
    assert getattr(a, "occ3_dsa_pin_forced", None) is None
    pin = dsa_pin.build(batch, "cpu", record=True)
    held, index = a.occ3_dsa_pin_forced
    assert held is pin[2] and index == 1
    pin[2][1] = 8
    assert dsa_pin.read_forced(a) == 8
    pin[2][1] = 10**6
    assert dsa_pin.read_forced(a) == 128
    assert dsa_pin.read_forced(ordinary()) is None
    assert dsa_pin.build(NS(reqs=[ordinary()]), "cpu", record=True) is None
    assert dsa_pin.build(NS(reqs=[]), "cpu", record=True) is None


def test_graph_pin_is_gone():
    """No CUDA graph carries the pin, so the static graph buffer is removed."""
    assert not hasattr(dsa_pin, "GraphPin")


def test_build_never_raises_and_records_a_fault(monkeypatch):
    """A pin that cannot be built leaves the batch unpinned and counts a fault."""
    before = dsa_pin.faults
    def boom(*a, **k):
        raise RuntimeError("synthetic")
    monkeypatch.setattr(dsa_pin, "_build", boom)
    assert dsa_pin.build(NS(reqs=[attached(0, 8)]), "cpu", record=True) is None
    assert dsa_pin.faults == before + 1


def test_note_fault_drops_the_batch_pin_and_logs_once(caplog):
    import logging
    caplog.set_level(logging.ERROR, logger="occ3_sglang.dsa_pin")
    saved = dsa_pin._fault_logged
    dsa_pin._fault_logged = False
    try:
        fb = NS(occ3_dsa_pin=("s", "c", "f"), occ3_dsa_pin_plan=("p",))
        before = dsa_pin.faults
        dsa_pin.note_fault(fb)
        dsa_pin.note_fault(None)
        assert fb.occ3_dsa_pin is None and fb.occ3_dsa_pin_plan is None
        assert dsa_pin.faults == before + 2
        assert sum("occ3 dsa pin fault" in r.getMessage() for r in caplog.records) == 1
    finally:
        dsa_pin._fault_logged = saved or dsa_pin._fault_logged


def test_short_attached_consume_carries_no_pin_and_keeps_graphs():
    """Opus r3: at or below index_topk the pin is a no-op, so nothing is pinned.

    build() returns None, so ForwardBatch.occ3_dsa_pin stays None and every
    graph runner's can_run_graph keeps the ordinary graph for the batch.
    """
    torch = pytest.importorskip("torch")
    short = attached(0, 8, prompt=1000)
    batch = NS(reqs=[ordinary(), short])
    assert dsa_pin.build(batch, "cpu", record=True) is None
    assert dsa_pin.build(batch, "cpu", record=False) is None
    assert getattr(short, "occ3_dsa_pin_forced", None) is None
    # An over-top-k consume in the same batch is pinned; the short one is not.
    long = attached(5, 8, prompt=4000)
    start, count, _ = dsa_pin.build(NS(reqs=[short, long]), "cpu", record=True)
    assert count.tolist() == [0, 8] and start.tolist() == [0, 5]


def test_pin_turns_on_at_the_crossing_step():
    """A receiver that grows past index_topk during decode is pinned from the
    first forward whose committed rows plus the margin exceed the top-k."""
    req = attached(0, 8, prompt=TOPK - 10)
    steps = []
    for _ in range(20):
        steps.append(dsa_pin.pin_rows([req], threshold=dsa_pin._threshold) is not None)
        req.output_ids.append(7)
    first = steps.index(True)
    assert not any(steps[:first]) and all(steps[first:])
    # Pinned exactly when prompt + output + margin > topk.
    assert (TOPK - 10) + first + MARGIN == TOPK + 1


def test_disarmed_threshold_pins_nothing():
    torch = pytest.importorskip("torch")
    dsa_pin.configure(None)
    assert dsa_pin.build(NS(reqs=[attached(0, 8, prompt=9000)]), "cpu", record=True) is None
    dsa_pin.configure(TOPK, 0)  # bad margin falls back to 1
    assert dsa_pin._threshold == (TOPK, 1)


def test_settle_evidence_rule_matches_the_producer():
    """Every request whose settle needs pin evidence was pinned on its last forward.

    _pin_evidence_required uses sequence_rows > topk; the producer pins at
    sequence_rows + margin > topk, which is weaker, so a requirement implies
    a pin (and a forced record). Short consumes need no evidence and get none.
    """
    from occ3_sglang import latent_worker

    runtime = NS(dsa_pin_active=True, sparse_index_topk=TOPK)
    for rows in range(TOPK - 6, TOPK + 6):
        req = attached(0, 8, prompt=rows)
        required = latent_worker._pin_evidence_required(runtime, req)
        pinned = dsa_pin.pin_rows([req], threshold=dsa_pin._threshold) is not None
        assert required == (rows > TOPK)
        assert not required or pinned
    short = attached(0, 8, prompt=100)
    assert latent_worker._pin_evidence_required(runtime, short) is False
    assert dsa_pin.pin_rows([short], threshold=dsa_pin._threshold) is None
