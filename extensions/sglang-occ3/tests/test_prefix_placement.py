"""Red-first: receiver continuous-prefix placement and marker framing.

The prefix must occupy reserved rows, never ordinary prompt rows, and the
receipt must say which layout produced the receiver sequence.
"""
from array import array
from types import SimpleNamespace as NS

import pytest

from occ3_sglang import latent_worker as w
from occ3_sglang import prefix_layout as pl
from occ3_sglang.latent_types import LatentError, RequestSlice
from test_latent_types import profile
from test_latent_registry import consume


class Tok:
    """Prefix-stable toy tokenizer: one id per character."""

    def __init__(self, unstable=False):
        self.unstable = unstable

    def encode(self, text, add_special_tokens=True):
        ids = [100 + (ord(ch) % 90) for ch in text]
        if self.unstable and text.endswith(">>"):
            ids[-1] = 999
        return ids


def req(descriptor=None, ids=(1, 2, 3)):
    return NS(rid="rid", output_ids=array("q"), origin_input_ids=array("q", list(ids)),
              origin_input_ids_unpadded=array("q", list(ids)),
              full_untruncated_fill_ids=array("q", list(ids)),
              extend_range=NS(end=len(ids)), session=None, multimodal_inputs=None,
              positional_embed_overrides=None, occ3_worker_descriptor=descriptor,
              retraction_count=0)


def admit_runtime(monkeypatch):
    record = NS(state="leased", admitted_consume=False, state_generation=123)
    runtime = NS(poisoned=False, resolve=lambda value, **kw: record,
                 preflight=lambda phase, check: check())
    monkeypatch.setattr(w, "_RUNTIME", runtime)
    return runtime, record


def test_marker_layout_places_prefix_after_the_marker():
    text = "system line\n" + pl.PREFIX_MARKER + "\nquestion"
    layout = pl.resolve_prefix_layout(text, Tok())
    assert layout.layout == pl.LAYOUT_MARKER
    assert layout.offset == text.index(pl.PREFIX_MARKER) + len(pl.PREFIX_MARKER)


def test_absent_marker_keeps_legacy_prepend_layout():
    layout = pl.resolve_prefix_layout("system line\nquestion", Tok())
    assert layout == pl.PrefixLayout(pl.LAYOUT_PREPEND, 0)


def test_repeated_marker_refuses():
    text = pl.PREFIX_MARKER + " a " + pl.PREFIX_MARKER
    with pytest.raises(LatentError):
        pl.resolve_prefix_layout(text, Tok())


def test_marker_at_position_zero_refuses():
    with pytest.raises(LatentError):
        pl.resolve_prefix_layout(pl.PREFIX_MARKER + " after", Tok())


def test_unstable_tokenisation_refuses_instead_of_guessing():
    with pytest.raises(LatentError):
        pl.resolve_prefix_layout("head " + pl.PREFIX_MARKER + " tail", Tok(unstable=True))


def test_prefix_span_reduces_to_legacy_window_at_offset_zero():
    for tokens, prefix_len, extend_len in ((8, 0, 4), (8, 0, 8), (8, 0, 32),
                                           (8, 4, 4), (8, 5, 8), (128, 0, 32768)):
        expected = max(0, min(extend_len, tokens - prefix_len))
        local, count, start = pl.prefix_span(tokens, 0, prefix_len, extend_len)
        assert (local, count, start) == (0, expected, prefix_len)
        assert w.prefix_overlap(tokens, prefix_len, extend_len) == expected


def test_prefix_span_moves_with_the_marker_offset():
    assert pl.prefix_span(8, 5, 0, 32) == (5, 8, 0)
    assert pl.prefix_span(8, 5, 0, 7) == (5, 2, 0)
    assert pl.prefix_span(8, 5, 6, 4) == (0, 4, 1)
    assert pl.prefix_span(8, 5, 20, 4) == (0, 0, 0)
    assert pl.prefix_span(8, 5, 0, 32, shift=1) == (4, 8, 0)


def test_admit_request_reserves_rows_at_the_marker_offset(monkeypatch):
    admit_runtime(monkeypatch)
    descriptor = consume(expected_prefix_tokens=4, prefix_offset=2, native_rid="rid")
    r = req(None, ids=(11, 12, 13))
    w.admit_request(NS(occ3_runtime=w._RUNTIME), r, descriptor)
    assert list(r.origin_input_ids) == [11, 12, 0, 0, 0, 0, 13]
    assert list(r.full_untruncated_fill_ids) == [11, 12, 0, 0, 0, 0, 13]


def test_admit_request_keeps_legacy_prepend_at_offset_zero(monkeypatch):
    admit_runtime(monkeypatch)
    descriptor = consume(expected_prefix_tokens=4, native_rid="rid")
    r = req(None, ids=(11, 12, 13))
    w.admit_request(NS(occ3_runtime=w._RUNTIME), r, descriptor)
    assert list(r.origin_input_ids) == [0, 0, 0, 0, 11, 12, 13]


def test_admit_request_refuses_offset_past_the_prompt(monkeypatch):
    admit_runtime(monkeypatch)
    descriptor = consume(expected_prefix_tokens=4, prefix_offset=4, native_rid="rid")
    with pytest.raises(LatentError):
        w.admit_request(NS(occ3_runtime=w._RUNTIME), req(None, ids=(11, 12, 13)), descriptor)


class Rows:
    is_cuda = True
    dtype = "bfloat16"

    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.streams = 0

    @property
    def shape(self):
        return (len(self.data), len(self.data[0]))

    def numel(self):
        return len(self.data) * len(self.data[0])

    def __getitem__(self, key):
        return Span(self, range(*key.indices(len(self.data))))

    def record_stream(self, stream):
        self.streams += 1


class Span:
    def __init__(self, rows, indices):
        self.rows, self.indices = rows, list(indices)

    def copy_(self, source):
        assert len(source.indices) == len(self.indices)
        for destination, origin in zip(self.indices, source.indices):
            self.rows.data[destination] = list(source.rows.data[origin])
        return self


def injection_runtime(monkeypatch, prefix):
    record = NS(prefix=prefix, events=[], live_refs=[], injected_positions=set(),
                capture_commit_id="commit")
    stream = NS(wait_event=lambda event: None)
    runtime = NS(poisoned=False, profile=profile(), device="cuda", resolve=lambda d: record,
                 preflight=lambda phase, check: check(),
                 torch=NS(cuda=NS(current_stream=lambda device: stream)), event=lambda: object())
    monkeypatch.setattr(w, "_RUNTIME", runtime)
    return record


def injection_batch(descriptor, start, length, prefix_len, hidden):
    slice_ = RequestSlice(descriptor, 0, length, prefix_len, 0, False)
    return NS(occ3_context=(slice_,), occ3_force_eager=True, batch_size=1,
              forward_mode=NS(is_extend=lambda: True, is_target_verify=lambda: False))


def test_injection_writes_only_the_reserved_marker_rows(monkeypatch):
    import sys, types
    module = types.ModuleType("sglang.srt.layers.layer_boundary")
    module.get_attn_tp_context = lambda: NS(input_scattered=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)
    hidden = profile().hidden_dim
    prefix = Rows([[10.0 + row] * 2 + [0.0] * (hidden - 2) for row in range(3)])
    descriptor = consume(expected_prefix_tokens=3, prefix_offset=2, native_rid="rid")
    record = injection_runtime(monkeypatch, prefix)
    prompt = Rows([[100.0 + row] * 2 + [0.0] * (hidden - 2) for row in range(7)])
    batch = injection_batch(descriptor, 0, 7, 0, hidden)
    w.inject_target_embeddings(batch, prompt)
    assert batch.occ3_effective_embeds == (2, 3)
    assert prompt.data[:2] == [[100.0] * 2 + [0.0] * (hidden - 2),
                               [101.0] * 2 + [0.0] * (hidden - 2)]
    assert prompt.data[2:5] == prefix.data
    assert prompt.data[5:] == [[105.0] * 2 + [0.0] * (hidden - 2),
                               [106.0] * 2 + [0.0] * (hidden - 2)]
    w.mark_injection(batch)
    assert record.injected_positions == {2, 3, 4}
    assert record.injection_commit_id


def test_adapter_resolves_marker_and_pins_token_ids():
    from types import SimpleNamespace as NS2
    from occ3_sglang.chat_adapter import ChatAttachmentAdapter
    adapter = ChatAttachmentAdapter()
    text = "sys\n" + pl.PREFIX_MARKER + "\nuser question"
    obj = NS2(text=text, input_ids=None)
    layout, pinned = adapter._prefix_layout(obj, NS2(tokenizer=Tok()))
    assert layout.layout == pl.LAYOUT_MARKER and pinned == Tok().encode(text)
    assert 0 < layout.offset < len(pinned)


def test_adapter_falls_back_to_prepend_without_marker_or_role_token():
    # No marker and no attestable user role token (char tokenizer): the
    # extension cannot place the prefix, so the legacy prepend layout runs
    # with a bounded fallback reason and the prompt is not re-pinned.
    from types import SimpleNamespace as NS2
    from occ3_sglang.chat_adapter import ChatAttachmentAdapter
    adapter = ChatAttachmentAdapter()
    layout, pinned = adapter._prefix_layout(NS2(text="plain prompt", input_ids=None),
                                            NS2(tokenizer=Tok()))
    assert (layout.layout, layout.offset, pinned) == (pl.LAYOUT_PREPEND, 0, None)
    assert layout.fallback in pl.FALLBACK_REASONS
    layout, pinned = adapter._prefix_layout(NS2(text="plain prompt", input_ids=None), NS2())
    assert (layout.layout, layout.fallback, pinned) == (pl.LAYOUT_PREPEND, "no_tokenizer", None)


def test_adapter_refuses_duplicate_marker():
    from types import SimpleNamespace as NS2
    from occ3_sglang.chat_adapter import AttachmentError, ChatAttachmentAdapter
    adapter = ChatAttachmentAdapter()
    text = "a " + pl.PREFIX_MARKER + " b " + pl.PREFIX_MARKER + " c"
    with pytest.raises(AttachmentError):
        adapter._prefix_layout(NS2(text=text, input_ids=None), NS2(tokenizer=Tok()))


def test_adapter_refuses_when_no_tokenizer_is_available():
    from types import SimpleNamespace as NS2
    from occ3_sglang.chat_adapter import AttachmentError, ChatAttachmentAdapter
    adapter = ChatAttachmentAdapter()
    with pytest.raises(AttachmentError):
        adapter._prefix_layout(NS2(text="x " + pl.PREFIX_MARKER + " y", input_ids=None), NS2())
