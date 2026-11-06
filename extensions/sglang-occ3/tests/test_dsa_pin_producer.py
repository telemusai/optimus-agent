"""DSA indexer pin producer and CUDA-graph static buffers (occ3_sglang.dsa_pin)."""
from types import SimpleNamespace as NS

import pytest

from occ3_sglang import dsa_pin
from test_latent_registry import consume
from test_latent_types import capture


def attached(start=0, count=8):
    return NS(occ3_attached=True,
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


def test_build_records_forced_on_target_only():
    torch = pytest.importorskip("torch")
    a = attached(3, 8)
    batch = NS(reqs=[ordinary(), a, ordinary()])
    start, count, forced = dsa_pin.build(batch, "cpu", record=False)
    assert start.dtype == count.dtype == forced.dtype == torch.int32
    assert start.tolist() == [0, 3, 0] and count.tolist() == [0, 8, 0]
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


def test_graph_pin_static_copy_and_padded_slot_zeroing():
    torch = pytest.importorskip("torch")
    graph = dsa_pin.GraphPin(4, "cpu")
    view = graph.view(4)
    # The captured view aliases the static storage.
    assert view[0].data_ptr() == graph.start.data_ptr()
    live = NS(occ3_dsa_pin=(torch.tensor([0, 7], dtype=torch.int32),
                            torch.tensor([0, 64], dtype=torch.int32),
                            torch.zeros(2, dtype=torch.int32)))
    graph.start.fill_(9); graph.count.fill_(9); graph.forced.fill_(9)
    graph.load(live, 2)
    assert graph.start.tolist() == [0, 7, 0, 0]
    assert graph.count.tolist() == [0, 64, 0, 0]
    assert graph.forced.tolist() == [0, 0, 0, 0]
    graph.forced[1] = 64
    graph.publish(live, 2)
    assert live.occ3_dsa_pin[2].tolist() == [0, 64]
    # Next replay without any attached row: every slot zeroed again.
    graph.load(NS(occ3_dsa_pin=None), 3)
    assert graph.count.tolist() == [0, 0, 0, 0] and graph.start.tolist() == [0, 0, 0, 0]
    graph.publish(NS(occ3_dsa_pin=None), 3)
    with pytest.raises(RuntimeError):
        graph.load(live, 5)
    assert view[1].data_ptr() == graph.count.data_ptr()


def test_graph_pin_load_is_graph_safe_source():
    import inspect
    text = inspect.getsource(dsa_pin.GraphPin)
    assert ".item()" not in text and ".tolist()" not in text and "cpu()" not in text
