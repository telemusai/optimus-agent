"""Task D: extension-side placement after the last user role token (LAYOUT_LAST_USER).

GLM-style toy tokenizer: the chat template's role tokens are single added
tokens; everything else is one id per character. Exact token-id match only.
"""
from array import array
from types import SimpleNamespace as NS

import pytest

from occ3_sglang import latent_worker as w
from occ3_sglang import prefix_layout as pl
from occ3_sglang import span_markers as sm
from occ3_sglang.attachment_contract import AttachmentError, ConsumeWorkerDescriptor
from occ3_sglang.chat_adapter import ChatAttachmentAdapter
from test_latent_registry import consume
from test_prefix_placement import (admit_runtime, injection_batch, injection_runtime, req, Rows)
from test_latent_types import profile

ADDED = {"[gMASK]": 154822, "<sop>": 154823, "<|system|>": 154824, "<|user|>": 154827,
         "<|assistant|>": 154828, "<|observation|>": 154829, "<think>": 154841,
         "</think>": 154842, "<|endoftext|>": 154820}
USER = ADDED["<|user|>"]


class GlmTok:
    vocab_size = 154856
    eos_token = "<|endoftext|>"
    eos_token_id = 154820

    def __init__(self, added=None, unstable=False):
        self.added = dict(ADDED if added is None else added)
        self.back = {v: k for k, v in self.added.items()}
        self.unstable = unstable
        self.calls = 0

    def __len__(self):
        return self.vocab_size

    def encode(self, text, add_special_tokens=True):
        self.calls += 1
        ids, i = [], 0
        while i < len(text):
            for literal, value in self.added.items():
                if text.startswith(literal, i):
                    ids.append(value)
                    i += len(literal)
                    break
            else:
                ids.append(1000 + ord(text[i]))
                i += 1
        if self.unstable and len(ids) > 1 and ids[-1] == USER:
            ids[-1] = 7  # head tokenises differently from the whole prompt
        return ids

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.back.get(v, chr(v - 1000) if v >= 1000 else "?") for v in ids)


def glm_prompt(turns, system="You are helpful."):
    out = "[gMASK]<sop><|system|>\n" + system
    for role, content in turns:
        out += "<|%s|>\n%s" % (role, content)
    return out + "<|assistant|><think>"


MULTI = glm_prompt([("user", "first question"), ("assistant", "an answer"),
                    ("user", "second question?")])


def adapter_layout(text=None, ids=None, tok=None):
    tok = GlmTok() if tok is None else tok
    return ChatAttachmentAdapter()._prefix_layout(NS(text=text, input_ids=ids), NS(tokenizer=tok))


def test_role_token_id_is_attested_from_the_tokenizer():
    assert sm.resolve_user_role_id(GlmTok()) == USER
    # A tokenizer without the single role token cannot attest it: no guess.
    missing = {k: v for k, v in ADDED.items() if k != "<|user|>"}
    assert sm.resolve_user_role_id(GlmTok(added=missing)) is None


def test_glm_prompt_places_prefix_after_the_last_user_role_token():
    tok = GlmTok()
    layout, pinned = adapter_layout(MULTI, tok=tok)
    ids = tok.encode(MULTI)
    last = max(i for i, v in enumerate(ids) if v == USER)
    assert layout == pl.PrefixLayout(pl.LAYOUT_LAST_USER, last + 1)
    assert pinned == ids and pinned[layout.offset - 1] == USER
    # Exactly the second user turn follows; earlier user turn is before it.
    assert tok.decode(pinned[layout.offset:]).startswith("\nsecond")
    assert sum(1 for v in pinned[:layout.offset] if v == USER) == 2


def test_role_literal_in_content_becomes_the_role_token_and_wins():
    # Real behaviour (HF tokenizer, special tokens enabled on the serving
    # path): a content literal "<|user|>" encodes to the role-token id, so the
    # prefix lands after it, inside that content. Documented limitation in
    # resolve_last_user_layout; it only moves the receiver's own state.
    tok = GlmTok()
    text = glm_prompt([("user", "real question"), ("assistant", "quote <|user|> here")])
    layout, pinned = adapter_layout(text, tok=tok)
    ids = tok.encode(text)
    literal = max(i for i, v in enumerate(ids) if v == USER)
    assert layout == pl.PrefixLayout(pl.LAYOUT_LAST_USER, literal + 1)
    assert tok.decode(pinned[layout.offset:]).startswith(" here")
    # Characters that only spell the literal (byte fallback ids) are not the
    # role token id: exact id match, never a text search.
    base = tok.encode(glm_prompt([("user", "q")]) + "x")
    fake = base[:-1] + [1000 + ord(c) for c in "<|user|>"] + base[-1:]
    layout, _ = adapter_layout(ids=fake, tok=tok)
    real = max(i for i, v in enumerate(fake) if v == USER)
    assert layout.layout == pl.LAYOUT_LAST_USER and layout.offset == real + 1


def test_tool_turns_after_the_last_user_turn_keep_the_well_before_them():
    tok = GlmTok()
    text = glm_prompt([("user", "look it up"), ("assistant", "calling tool"),
                       ("observation", "tool result"), ("assistant", "more"),
                       ("observation", "second result")])
    layout, pinned = adapter_layout(text, tok=tok)
    ids = tok.encode(text)
    user = max(i for i, v in enumerate(ids) if v == USER)
    assert layout == pl.PrefixLayout(pl.LAYOUT_LAST_USER, user + 1)
    tail = pinned[layout.offset:]
    assert tok.decode(tail).startswith("\nlook it up")
    assert tail.count(ADDED["<|observation|>"]) == 2


def test_find_last_single_id_matches_the_general_scan():
    for hay in ([], [5], [5, 1, 5, 2], [1, 2, 3], list(range(50)) * 3):
        for needle in (5, 2, 49, 99):
            general = next((i for i in range(len(hay) - 1, -1, -1) if hay[i] == needle), -1)
            assert pl._find_last(hay, [needle]) == general
    assert pl._find_last([1, 2, 1, 2, 3], [1, 2]) == 2


def test_prepend_fallback_logs_one_bounded_info_line():
    import inspect

    from occ3_sglang import chat_adapter

    missing = {k: v for k, v in ADDED.items() if k != "<|user|>"}
    layout, _ = adapter_layout(MULTI, tok=GlmTok(added=missing))
    assert layout.fallback == "no_role_token"
    # Source check of the admit site: one info line, reason from the closed set.
    source = inspect.getsource(chat_adapter.ChatAttachmentAdapter)
    assert 'logger.info("occ3 consume prefix layout prepend fallback=%s",' in source
    assert source.count("prepend fallback=%s") == 1


def test_explicit_marker_still_wins():
    text = MULTI.replace("second", pl.PREFIX_MARKER + "second")
    layout, pinned = adapter_layout(text)
    assert layout.layout == pl.LAYOUT_MARKER
    assert pinned[layout.offset - 1] != USER  # after the marker, not the role token


def test_token_id_prompts_need_no_text_proof():
    tok = GlmTok()
    ids = tok.encode(MULTI)
    layout, pinned = adapter_layout(ids=list(ids), tok=GlmTok())
    assert layout.layout == pl.LAYOUT_LAST_USER and pinned == ids


@pytest.mark.parametrize("case,reason", [
    ("no_role_token", "no_role_token"), ("role_absent", "role_absent"),
    ("unstable", "unstable"), ("role_last", "shape"), ("empty", "shape")])
def test_fallbacks_use_prepend_and_name_the_reason(case, reason):
    tok, text, ids = GlmTok(), MULTI, None
    if case == "no_role_token":
        tok = GlmTok(added={k: v for k, v in ADDED.items() if k != "<|user|>"})
    elif case == "role_absent":
        text = glm_prompt([], system="only a system turn")
    elif case == "unstable":
        tok = GlmTok(unstable=True)
    elif case == "role_last":
        text = "[gMASK]<sop><|user|>"
    elif case == "empty":
        text, ids = None, None
    layout, pinned = adapter_layout(text, ids, tok=tok)
    assert layout == pl.PrefixLayout(pl.LAYOUT_PREPEND, 0, fallback=reason)
    assert pinned is None


def test_layout_invariants():
    with pytest.raises(Exception):
        pl.PrefixLayout(pl.LAYOUT_LAST_USER, 0)
    with pytest.raises(Exception):
        pl.PrefixLayout(pl.LAYOUT_LAST_USER, 5, fallback="unstable")
    with pytest.raises(Exception):
        pl.PrefixLayout(pl.LAYOUT_PREPEND, 0, fallback="free text")


def test_descriptor_carries_layout_and_offset():
    d = consume(prefix_offset=42, prefix_layout="last_user")
    assert (d.prefix_offset, d.prefix_layout) == (42, "last_user")
    for bad in (dict(prefix_offset=0, prefix_layout="last_user"),
                dict(prefix_offset=3, prefix_layout="prepend"),
                dict(prefix_offset=3, prefix_layout="other")):
        with pytest.raises(AttachmentError):
            consume(**bad)
    from occ3_sglang.attachment_contract import normalize_worker_descriptor
    from dataclasses import asdict
    assert normalize_worker_descriptor(asdict(d)) == d


def test_pin_start_follows_last_user_offset():
    from occ3_sglang import dsa_pin
    tok = GlmTok()
    layout, pinned = adapter_layout(MULTI, tok=tok)
    d = consume(prefix_offset=layout.offset, prefix_layout=layout.layout, expected_prefix_tokens=16)
    rows = dsa_pin.pin_rows([NS(occ3_attached=True, occ3_worker_descriptor=d)])
    assert rows == ([layout.offset], [16])


def test_admit_reserves_rows_after_the_role_token_and_records_layout(monkeypatch):
    runtime, record = admit_runtime(monkeypatch)
    seen = {}
    runtime.preflight = lambda phase, check: seen.setdefault(phase, check())
    tok = GlmTok()
    layout, pinned = adapter_layout(MULTI, tok=tok)
    d = consume(expected_prefix_tokens=4, prefix_offset=layout.offset,
                prefix_layout=layout.layout, native_rid="rid")
    r = req(None, ids=pinned)
    w.admit_request(NS(occ3_runtime=runtime), r, d)
    got = list(r.origin_input_ids)
    assert got[:layout.offset] == pinned[:layout.offset] and got[layout.offset - 1] == USER
    assert got[layout.offset:layout.offset + 4] == [0, 0, 0, 0]
    assert got[layout.offset + 4:] == pinned[layout.offset:]
    assert seen["admit"]["prefix_layout"] == "last_user"
    assert seen["admit"]["prefix_offset"] == layout.offset


def _inject_mod(monkeypatch):
    import sys, types
    module = types.ModuleType("sglang.srt.layers.layer_boundary")
    module.get_attn_tp_context = lambda: NS(input_scattered=False)
    monkeypatch.setitem(sys.modules, module.__name__, module)


@pytest.mark.parametrize("chunks", [
    [(0, 30)],            # one prefill
    [(0, 20), (20, 10)],  # prefix wholly in the last chunk
    [(0, 23), (23, 7)],   # prefix straddles the chunk boundary (offset 21)
    [(0, 21), (21, 9)],   # boundary exactly at the offset
    [(0, 25), (25, 5)],   # last chunk holds only the tail of the prefix
])
def test_chunked_prefill_with_prefix_near_the_end(monkeypatch, chunks):
    """Prompt 26 rows, offset 21 (last user turn near the end), prefix 4 rows."""
    _inject_mod(monkeypatch)
    hidden = profile().hidden_dim
    offset, count, total = 21, 4, 30
    prefix = Rows([[10.0 + row] * 2 + [0.0] * (hidden - 2) for row in range(count)])
    d = consume(expected_prefix_tokens=count, prefix_offset=offset,
                prefix_layout="last_user", native_rid="rid")
    record = injection_runtime(monkeypatch, prefix)
    record.state = "leased"
    full = [[100.0 + row] * 2 + [0.0] * (hidden - 2) for row in range(total)]
    out = []
    for start, length in chunks:
        prompt = Rows(full[start:start + length])
        batch = injection_batch(d, 0, length, start, hidden)
        w.inject_target_embeddings(batch, prompt)
        span = batch.occ3_effective_embeds
        if span is None:
            assert not (start < offset + count and offset < start + length)
        else:
            assert span[0] == max(start, offset)
        w.mark_injection(batch)
        out.extend(prompt.data)
    assert out[:offset] == full[:offset] and out[offset + count:] == full[offset + count:]
    assert out[offset:offset + count] == prefix.data
    assert record.injected_positions == set(range(offset, offset + count))
    assert record.state == "injected" and record.injection_commit_id
    # Cohort release is keyed on injection completeness, not on the layout.
    record.consumer = d
    runtime = w._RUNTIME
    runtime.registry = NS(source_for_attempt=lambda attempt: record)
    r = NS(occ3_worker_descriptor=d, occ3_force_eager=True)
    assert w.release_consume_cohort(NS(occ3_runtime=runtime), r) is True
    assert r.occ3_force_eager is False


def test_cohort_not_released_before_the_last_chunk_injects(monkeypatch):
    _inject_mod(monkeypatch)
    hidden = profile().hidden_dim
    prefix = Rows([[1.0] * hidden for _ in range(4)])
    d = consume(expected_prefix_tokens=4, prefix_offset=21, prefix_layout="last_user", native_rid="rid")
    record = injection_runtime(monkeypatch, prefix)
    record.state, record.consumer = "leased", d
    batch = injection_batch(d, 0, 23, 0, hidden)  # first chunk covers rows 21,22 only
    w.inject_target_embeddings(batch, Rows([[0.0] * hidden for _ in range(23)]))
    w.mark_injection(batch)
    assert record.injected_positions == {21, 22} and record.state != "injected"
    w._RUNTIME.registry = NS(source_for_attempt=lambda attempt: record)
    r = NS(occ3_worker_descriptor=d, occ3_force_eager=True)
    assert w.release_consume_cohort(NS(occ3_runtime=w._RUNTIME), r) is False
    assert r.occ3_force_eager is True



# (a) MTP/EAGLE draft seed: draft-extend publishes indexer top-k rows at
# select = cumsum(extend_seq_lens) - 1 (the last prompt row of each request),
# and write_mtp_seed copies topk_indices[select] whole into a buffer whose
# width must equal the top-k width (copy_ of [n, topk] into [n, width]). The
# unfused/paged export is a passthrough (test_dsa_mtp_seed_patch). So the seed
# carries the pinned range iff the indexer selection at the last prompt row
# carries it, which this checks for a prefix that sits near the end.
from test_dsa_pin_mask import indexer, _pin, _select  # noqa: E402,F401


@pytest.mark.parametrize("tail", [1, 6, 40])
def test_draft_seed_keeps_a_pin_near_the_end_of_the_prompt(indexer, tail):
    torch = indexer.torch
    length, count, topk = 3000, 16, 2048
    offset = length - count - tail
    rows = 8  # last prefill chunk rows of one request (ragged, ks = 0)
    lengths = torch.arange(length - rows + 1, length + 1, dtype=torch.int32)
    logits = torch.randn(rows, length) - 5.0
    pin = _pin(torch, [offset], [count])
    indexer.mask(logits, lengths, torch.zeros(rows, dtype=torch.int32), pin=pin,
                 row_batch=torch.zeros(rows, dtype=torch.int32))
    sel = _select(torch, logits, lengths, torch.zeros(rows, dtype=torch.int32), topk)
    select = torch.tensor([rows - 1])  # cumsum(extend_seq_lens) - 1
    seed = sel[select]
    assert set(range(offset, offset + count)) <= set(seed[0].tolist())
    assert pin[2].tolist() == [count]
    # Earlier rows of the chunk only see the causally visible part.
    first_len = int(lengths[0])
    visible = max(0, min(count, first_len - offset))
    assert set(range(offset, offset + visible)) <= set(sel[0].tolist())
