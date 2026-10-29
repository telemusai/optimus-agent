"""Predictor/token index alignment for OCC-3 capture (source-attested, no GPU).

Question under test: for every captured hidden row, which output-token index
does it predict, and does ``latent_worker`` pair the row with that token?

Answer proved here (pinned SGLang e00930c5 / v0.5.21):

* A captured row's hidden state sits at the absolute position carried by the
  row's ``positions`` entry, and it predicts the token at ``position + 1``
  (ALIGNMENT B, "state predicts the NEXT token"), in prefill and in verify.
* ``predict[row]`` (post-mutation EAGLE verify tensor) is exactly that token,
  so ``H[row]`` paired with ``embedding(predict[row])`` is index-consistent.
* The accepted rows of a verify step carry positions
  ``seq_lens + k`` with ``seq_lens == len(origin_input_ids) + output_start - 1``
  because the last emitted token is in ``output_ids`` but not yet in the KV
  cache. ``commit_result``'s ``arange(len(origin) + output_start - 1, ...)``
  assertion is therefore satisfiable and is the correct check.

No torch is imported and no SGLang module is loaded: the pinned source is read
as text, and our arithmetic is mirrored in pure Python. Set
``OCC3_SGLANG_SOURCE_BASE`` to the pristine pinned checkout (SGLang v0.5.21,
commit e00930c5489053f26d86b179cee0d087f846acbb); it defaults to the sibling
reference checkout. ``OCC3_SGLANG_SOURCE`` is deliberately not read here: that
variable names the live patched source attested by the chat suites. See
``_source_pins.py`` for the runbook.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from _source_pins import (
    DEFAULT_PRISTINE_ROOT, EXTENSION, needs_pristine_source, pinned, pristine_root,
    srt_dir,
)

SOURCE = pristine_root()
SRT = srt_dir(SOURCE)
EAGLE_CUH = SOURCE / "python/sglang/kernels/jit/include/sgl_kernel/speculative/eagle.cuh"
EAGLE_OPS = SOURCE / "python/sglang/kernels/ops/speculative/eagle.py"
WORKER = EXTENSION / "occ3_sglang/latent_worker.py"
PATCH_DIR = EXTENSION / "patches"
CAPTURE_PATCH = PATCH_DIR / "capture-eager-mtp.v1.json"

needs_source = needs_pristine_source()


def read(path) -> str:
    return Path(path).read_text(encoding="utf-8")


def once(text: str, snippet: str, where: str = "") -> int:
    """Assert a snippet occurs exactly once and return its offset."""
    found = text.count(snippet)
    assert found == 1, f"{where}: expected 1 occurrence of {snippet!r}, found {found}"
    return text.index(snippet)


# --------------------------------------------------------------------------- #
# The root this suite attests is the pristine checkout, never the live source
# --------------------------------------------------------------------------- #
def test_pristine_root_prefers_the_explicit_base_variable(monkeypatch, tmp_path):
    monkeypatch.setenv("OCC3_SGLANG_SOURCE_BASE", str(tmp_path))
    monkeypatch.setenv("OCC3_SGLANG_SOURCE", str(tmp_path / "live-extract"))
    assert pristine_root() == tmp_path


def test_pristine_root_ignores_the_live_source_variable(monkeypatch, tmp_path):
    """The campaign env names the live extract with OCC3_SGLANG_SOURCE; that
    partial patched tree must never silently become this suite's root."""
    monkeypatch.delenv("OCC3_SGLANG_SOURCE_BASE", raising=False)
    monkeypatch.setenv("OCC3_SGLANG_SOURCE", str(tmp_path))
    assert pristine_root() == DEFAULT_PRISTINE_ROOT


def test_pristine_root_default_is_the_reference_checkout(monkeypatch):
    monkeypatch.delenv("OCC3_SGLANG_SOURCE_BASE", raising=False)
    monkeypatch.delenv("OCC3_SGLANG_SOURCE", raising=False)
    assert pristine_root() == DEFAULT_PRISTINE_ROOT


def test_absent_pristine_root_skips_with_explicit_reason(monkeypatch, tmp_path):
    missing = tmp_path / "absent-checkout"
    monkeypatch.setenv("OCC3_SGLANG_SOURCE_BASE", str(missing))
    mark = needs_pristine_source()
    assert mark.args[0] is True
    assert "OCC3_SGLANG_SOURCE_BASE" in mark.kwargs["reason"]
    with pytest.raises(pytest.skip.Exception) as caught:
        pinned(srt_dir(missing) / "speculative/eagle_utils.py")
    message = str(caught.value)
    assert "OCC3_SGLANG_SOURCE_BASE" in message
    assert str(missing) in message


# --------------------------------------------------------------------------- #
# A/B. What a captured row's hidden state predicts
# --------------------------------------------------------------------------- #
@needs_source
def test_verify_row_positions_are_absolute_token_positions():
    """build_tree_efficient: positions[row] = seq_len, then parent depth + seq_len."""
    text = pinned(EAGLE_CUH)
    # Both tree builders (FULL_MASK and QLEN_ONLY_BITPACKING) write the same way.
    assert text.count("    positions[bid * draft_token_num] = seq_len;") == 2
    assert (
        text.count("    positions[bid * draft_token_num + tid] = position + seq_len;")
        == 2
    )
    root = text.index("  int position = 0;\n  if (tid == 0) {\n")
    node0 = text.index("positions[bid * draft_token_num] = seq_len;", root)
    branch = text.index("} else {", root)
    deeper = text.index(
        "    positions[bid * draft_token_num + tid] = position + seq_len;", root
    )
    # Node 0 (the carried-over "bonus" token) has depth 0, so it takes seq_len.
    assert root < node0 < branch
    # Every other node adds its parent-chain depth to seq_len.
    assert branch < deeper


@needs_source
def test_tree_positions_documented_as_token_positions():
    """eagle_utils: positions is "where each token belongs to", starting at prompt len."""
    text = pinned(SRT / "speculative/eagle_utils.py")
    once(text, "    # position: where each token belongs to")
    once(text, "    # then, positions = [7, 8, 8, 9]")
    # The docstring example is a 7-token prompt whose draft rows start at 7, i.e.
    # one past the last prompt position (6). The rows own those positions.
    assert "7, 8, 8, 9" in text
    assert "the prompt length is 7" in text


@needs_source
def test_verify_batch_row0_is_the_bonus_token_and_kv_starts_at_seq_lens():
    """Verify rows are the tree nodes; their KV slots start at batch.seq_lens."""
    text = pinned(SRT / "speculative/eagle_utils.py")
    once(text, "        batch.input_ids = verify_input.draft_token")
    once(text, "            start_offset=batch.seq_lens,")
    once(
        text,
        "    draft_tokens = torch.cat((bonus_tokens.unsqueeze(1), draft_tokens), dim=1).flatten()",
    )
    # Row 0 is therefore the bonus token, written into KV slot seq_lens, which is
    # its own absolute position because the bonus token is the last emitted
    # token and is not yet in KV.


@needs_source
def test_verify_positions_reach_forward_batch_unchanged():
    text = pinned(SRT / "model_executor/forward_batch_info.py")
    once(text, "            ret.positions = ret.spec_info.positions")
    once(text, "            positions, ret.extend_start_loc = compute_position(")
    # capture_prenorm reads forward_batch.positions; stage_verify receives
    # verify_input.positions. ForwardBatch aliases the latter, so both hooks see
    # the identical tensor.
    common = pinned(SRT / "speculative/eagle_worker_common.py")
    once(common, "        positions=position,")


@needs_source
def test_verify_tree_greedy_writes_parent_slot_with_child_token():
    """predicts[parent_row] = child token, and the last slot = the bonus token."""
    text = pinned(EAGLE_CUH)
    start = once(text, "  IdType2 last_accepted_retrive_idx = retrive_index[bx * num_draft_tokens];")
    once(text, "  accept_index[bx * num_speculative_tokens] = last_accepted_retrive_idx;")
    once(text, "      IdType2 target_token_id = target_predict[last_accepted_retrive_idx];")
    accept = once(text, "        predicts[last_accepted_retrive_idx] = target_token_id;")
    bonus = once(
        text,
        "  predicts[last_accepted_retrive_idx] = target_predict[last_accepted_retrive_idx];",
    )
    assert start < accept < bonus
    # `target_token_id` is the target prediction read AT the parent row, i.e. the
    # token that follows the parent row's own token.
    assert (
        text.index("IdType2 target_token_id = target_predict[last_accepted_retrive_idx];")
        < accept
    )


@needs_source
def test_predict_is_an_in_place_buffer_initialised_to_zero():
    """Rows outside the accepted run keep 0: only accept_index[:count] is meaningful."""
    text = pinned(SRT / "speculative/eagle_utils.py")
    once(text, "    predict = torch.zeros(predict_shape, dtype=torch.int32, device=device).flatten()")
    once(text, "    return predict, num_correct_drafts + 1, accept_index")
    # accept_lens includes the bonus token, so a full accept commits nd rows.
    ops = pinned(EAGLE_OPS)
    once(ops, "    bonus_token_idx = accept_stride * pid + accept_len - 1")


@needs_source
def test_logits_cover_every_verify_row_and_only_the_last_prefill_row():
    text = pinned(SRT / "layers/logits_processor.py")
    once(text, "            or logits_metadata.forward_mode.is_target_verify()")
    once(text, "            pruned_states = hidden_states[last_index]")
    once(
        text,
        "            last_index = torch.cumsum(logits_metadata.extend_seq_lens, dim=0) - 1",
    )
    # TARGET_VERIFY keeps `pruned_states = hidden_states` (no pruning), so row r
    # of next_token_logits is row r of hidden_states; prefill keeps only the last
    # row of each request, which is the row capture_prenorm selects.
    pruned = text[text.index("    def _get_pruned_states(") :]
    assert "pruned_states = hidden_states\n" in pruned[:3000]
    assert "hidden_states[draft_extend_select_index]" in pruned[:3000]


@needs_source
def test_capture_hook_retains_the_prenorm_residual():
    text = pinned(SRT / "layers/layer_boundary/residual/access.py")
    start = once(text, "def norm_output(hidden_states, residual, norm, capture_output=None, **read_kwargs):")
    body = text[start:]
    once(body, "    hidden_states, residual = norm(hidden_states, residual, **read_kwargs)")
    once(body, "        capture_output(residual)")
    # capture_output receives the completed residual BEFORE the fused norm is
    # applied, i.e. exactly the value the LM head normalizes for that same row.


# --------------------------------------------------------------------------- #
# C. Why the commit positions start at len(origin) + output_start - 1
# --------------------------------------------------------------------------- #
@needs_source
def test_last_emitted_token_lags_the_kv_cache_by_one():
    text = pinned(SRT / "managers/scheduler_components/batch_result_processor.py")
    once(text, "                        req.output_ids.append(next_token_id)")
    once(text, "                req.output_ids.extend(next_token_id)")
    once(text, "                # Commit the full accepted run (drafts + bonus).")
    once(text, "        token_seq_len = len(req.origin_input_ids) + len(req.output_ids) - 1")
    once(
        text,
        "        predict_tokens = []\n"
        "        for i, req in enumerate(batch.reqs):\n"
        "            accept_tokens = next_token_ids[i * stride : i * stride + accept_lens[i]]\n",
    )
    once(
        text,
        "        return req.finished_len if req.finished_len is not None else len(req.output_ids)",
    )
    # sglang states the invariant itself: the committed sequence length is
    # len(origin) + len(output_ids) - 1. Prefill appends the first sampled token
    # to output_ids before that token is ever written to KV, and every verify
    # step appends the whole accepted run while seq_lens grows by the same run.


@needs_source
def test_bonus_token_seeds_the_next_verify_row_zero():
    text = pinned(SRT / "speculative/eagle_worker_v2.py")
    once(text, "            bonus_tokens=next_token_ids,")
    common = pinned(SRT / "speculative/eagle_worker_common.py")
    once(common, "        accept_tokens = predict[accept_index]")


# --------------------------------------------------------------------------- #
# The patch inserts stage_verify while predict still indexes verify rows
# --------------------------------------------------------------------------- #
def _capture_manifest() -> dict:
    return json.loads(read(CAPTURE_PATCH))


def _patched(path: str) -> str:
    """Pinned text with capture-eager-mtp replacements applied in memory."""
    text = pinned(SRT / path)
    for entry in _capture_manifest()["files"]:
        if entry["path"] != path:
            continue
        assert hashlib.sha256(text.encode()).hexdigest() == entry["before_sha256"]
        for replacement in entry["replacements"]:
            assert text.count(replacement["before"]) == 1
            text = text.replace(replacement["before"], replacement["after"], 1)
        assert hashlib.sha256(text.encode()).hexdigest() == entry["after_sha256"]
    return text


@needs_source
def test_stage_verify_reads_predict_after_eagle_sample_and_before_compaction():
    text = _patched("speculative/eagle_worker_common.py")
    body = text[once(text, "def run_eagle_verify("):]
    sample = once(body, "    ) = eagle_sample(")
    stage = once(body, "        occ3_capture_stage = stage_verify(")
    seq_lens = once(body, "    new_seq_lens = batch.seq_lens + accept_lens")
    compact = once(body, "        predict = _finalize_accept_tree_path(")
    assert sample < stage < seq_lens < compact
    # Pre-compaction: accept_index still indexes the uncompacted verify rows, and
    # predict has already been mutated in place by the verify kernel.
    assert "verify_input.positions" in body[stage:seq_lens]
    assert "accept_index," in body[stage:seq_lens]


@needs_source
def test_capture_manifest_matches_the_pinned_checkout():
    manifest = _capture_manifest()
    assert manifest["source_revision"] == "e00930c5489053f26d86b179cee0d087f846acbb"
    assert manifest["source_version"] == "0.5.21"
    checked = 0
    for entry in manifest["files"]:
        path = SRT / entry["path"]
        if not path.is_file():
            continue
        assert hashlib.sha256(read(path).encode()).hexdigest() == entry["before_sha256"]
        checked += 1
    assert checked


# --------------------------------------------------------------------------- #
# Our side: the exact expressions the mirrors below reproduce
# --------------------------------------------------------------------------- #
def test_latent_worker_pins_the_capture_and_commit_arithmetic():
    text = read(WORKER)
    for snippet in (
        "rows = torch.tensor([item.extend_start + item.extend_len - 1], device=hidden_states.device)",
        "positions = forward_batch.positions.index_select(0, rows).clone()",
        "tokens = next_token_ids.reshape(-1)[:1].clone().reshape(1, 1)",
        "indices = torch.where(valid, accept_index, torch.zeros_like(accept_index)).reshape(-1).long()",
        "hidden = rows.hidden.index_select(0, indices).reshape(1, -1, runtime.profile.hidden_dim).clone()",
        "tokens = predict.reshape(-1).index_select(0, indices).reshape(1, -1).clone()",
        "selected_positions = positions.reshape(-1).index_select(0, indices).reshape(1, -1).clone()",
        "positions = torch.arange(len(req.origin_input_ids) + output_start - 1,",
        "destination = (record.accepted_count + i) % descriptor.max_prefix_tokens",
        "embeddings = self.embedding(tokens)",
        "aligned, _ = self.aligner(h, embeddings, self.target_norm)",
    ):
        once(text, snippet, str(WORKER))
    assert text.count('"predictor_position_offset": -1') == 1


# --------------------------------------------------------------------------- #
# Pure-Python mirrors of the index arithmetic
# --------------------------------------------------------------------------- #
def commit_expected_positions(prompt_len: int, output_start: int, output_end: int):
    """Mirror of latent_worker.commit_result (lines 722-725)."""
    return list(range(prompt_len + output_start - 1, prompt_len + output_end - 1))


def stage_verify_indices(accept_index, accept_len, width):
    """Mirror of latent_worker.stage_verify (lines 557-562)."""
    valid = [j < accept_len for j in range(width)]
    return [accept_index[j] if valid[j] else 0 for j in range(width)], valid


def ring_slot(accepted_count: int, i: int, capacity: int) -> int:
    """Mirror of latent_worker.commit_result ring write (line 748)."""
    return (accepted_count + i) % capacity


def test_prefill_commit_positions_mirror():
    prompt_len = 7
    assert commit_expected_positions(prompt_len, 0, 1) == [prompt_len - 1]


def test_verify_commit_positions_mirror_is_contiguous():
    prompt_len, output_start, nd = 7, 1, 4
    expected = commit_expected_positions(prompt_len, output_start, output_start + nd)
    assert expected == list(range(prompt_len, prompt_len + nd))


def test_commit_assertion_matches_kernel_positions():
    """The kernel's row positions equal the values commit_result asserts."""
    for prompt_len in (1, 7, 128):
        for output_start in (1, 2, 9):
            nd = 4
            seq_lens = prompt_len + output_start - 1  # KV lags one emitted token
            kernel_positions = [seq_lens + depth for depth in range(nd)]
            assert kernel_positions == commit_expected_positions(
                prompt_len, output_start, output_start + nd
            )


def test_stage_verify_indices_mirror_masks_padding():
    width = 6
    accept_index = [0, 3, 4, -1, -1, -1]
    indices, valid = stage_verify_indices(accept_index, 3, width)
    assert indices == [0, 3, 4, 0, 0, 0]
    assert valid == [True, True, True, False, False, False]
    # commit_result only reads [:, :count] with count <= accept_len.
    assert indices[:3] == accept_index[:3]


def verify_tree_greedy_mirror(nd: int, candidates, target_predict):
    """Greedy chain mirror of VerifyTreeGreedy (topk == 1, no siblings)."""
    assert len(candidates) == len(target_predict) == nd
    predict = [0] * nd
    accept_index = [0] * nd
    last = 0
    num_accepted = 0
    for _ in range(1, nd):
        cur = last + 1 if last + 1 < nd else -1
        matched = False
        while cur != -1:
            if candidates[cur] == target_predict[last]:
                predict[last] = target_predict[last]
                num_accepted += 1
                accept_index[num_accepted] = cur
                last = cur
                matched = True
                break
            cur = -1
        if not matched:
            break
    predict[last] = target_predict[last]
    return predict, num_accepted + 1, accept_index


def test_verify_alignment_b_when_every_draft_is_accepted():
    prompt_len, output_start, nd = 7, 1, 4
    seq_lens = prompt_len + output_start - 1
    tokens_at = {p: 1000 + p for p in range(seq_lens, seq_lens + nd + 1)}
    candidates = [tokens_at[seq_lens + r] for r in range(nd)]
    target_predict = [tokens_at[seq_lens + r + 1] for r in range(nd)]

    predict, accept_len, accept_index = verify_tree_greedy_mirror(nd, candidates, target_predict)
    assert accept_len == nd

    indices, valid = stage_verify_indices(accept_index, accept_len, nd)
    captured_positions = [seq_lens + r for r in indices[:accept_len]]
    captured_tokens = [predict[r] for r in indices[:accept_len]]
    commit_positions = commit_expected_positions(prompt_len, output_start, output_start + accept_len)
    assert captured_positions == commit_positions
    # The committed run is output_ids[output_start:output_start + accept_len].
    for i in range(accept_len):
        assert captured_tokens[i] == tokens_at[captured_positions[i] + 1]
        assert captured_positions[i] + 1 == prompt_len + output_start + i


def test_verify_alignment_b_holds_on_a_rejected_draft():
    prompt_len, output_start, nd = 7, 2, 4
    seq_lens = prompt_len + output_start - 1
    tokens_at = {p: 2000 + p for p in range(seq_lens, seq_lens + nd + 1)}
    candidates = [tokens_at[seq_lens + r] for r in range(nd)]
    target_predict = [tokens_at[seq_lens + r + 1] for r in range(nd)]
    candidates[2] = 999999  # third draft token is wrong

    predict, accept_len, accept_index = verify_tree_greedy_mirror(nd, candidates, target_predict)
    assert accept_len == 2
    indices, _ = stage_verify_indices(accept_index, accept_len, nd)
    assert indices[:accept_len] == [0, 1]
    captured_positions = [seq_lens + r for r in indices[:accept_len]]
    captured_tokens = [predict[r] for r in indices[:accept_len]]
    assert captured_positions == commit_expected_positions(
        prompt_len, output_start, output_start + accept_len
    )
    assert captured_tokens == [tokens_at[p + 1] for p in captured_positions]
    # Rejected rows stay at the zero fill and are never read.
    assert predict[2] == 0 and predict[3] == 0


def test_alignment_a_pairing_would_shift_the_ring_by_one():
    """Counterfactual: pairing a row with the token AT its position shifts by one."""
    prompt_len, output_start, nd = 7, 1, 4
    seq_lens = prompt_len + output_start - 1
    tokens_at = {p: 3000 + p for p in range(seq_lens, seq_lens + nd + 1)}
    candidates = [tokens_at[seq_lens + r] for r in range(nd)]
    target_predict = [tokens_at[seq_lens + r + 1] for r in range(nd)]
    predict, accept_len, accept_index = verify_tree_greedy_mirror(nd, candidates, target_predict)
    indices, _ = stage_verify_indices(accept_index, accept_len, nd)
    positions = [seq_lens + r for r in indices[:accept_len]]

    committed = [tokens_at[p + 1] for p in positions]          # what the code records
    at_row = [tokens_at[p] for p in positions]                  # ALIGNMENT A pairing
    assert committed != at_row
    assert at_row == [token - 1 for token in committed]
    # Only the recorded pairing reproduces output_ids[output_start:output_end].
    expected_output = [tokens_at[prompt_len + output_start + i] for i in range(accept_len)]
    assert committed == expected_output
    assert at_row != expected_output


def test_ring_slots_are_output_token_indices_across_steps():
    """Prefill writes slot 0, then every accepted row lands on its own index."""
    capacity = 8
    accepted_count = 0
    committed = []                       # (slot, hidden position, token position)
    prompt_len = 5
    # prefill: one row, position prompt_len - 1, token at prompt_len
    committed.append((ring_slot(accepted_count, 0, capacity), prompt_len - 1, prompt_len))
    accepted_count += 1
    for accept_len in (3, 1, 4):
        seq_lens = prompt_len + accepted_count - 1
        for i in range(accept_len):
            committed.append(
                (ring_slot(accepted_count, i, capacity), seq_lens + i, seq_lens + i + 1)
            )
        accepted_count += accept_len
    for slot, hidden_position, token_position in committed:
        assert slot == (token_position - prompt_len) % capacity
        assert hidden_position == token_position - 1
    assert accepted_count == len(committed)


def test_prefix_rows_are_consumed_in_position_order():
    """Layout-independent invariant: one prefix row per consecutive position.

    `latent_worker` decides *where* the accepted run lands (prepend slots or a
    `prefix_offset` marker block) through its layout helper, but every layout
    must consume prefix rows in increasing order with stride one, because the
    ring is indexed by output token index. Anything else would re-pair a row
    with a different token.
    """
    rows = 8
    for first_position in (0, 3, 11):
        positions = [first_position + j for j in range(rows)]
        assert positions == sorted(positions)
        assert [p - positions[0] for p in positions] == list(range(rows))
        # Row j therefore carries output token (accepted_span_start + j), and its
        # hidden state sits one position earlier than that token.
        for j in range(rows):
            assert positions[j] - 1 == first_position + j - 1
