"""Offline tests for the tokenizer-attested marker table.

No SGLang, no Torch, no GPU, no network, no real tokenizer.  The fake tokenizer
below mirrors the two properties attestation depends on: one exact id per
added-token spelling, and a byte fallback that ``decode`` reverses exactly, the
way a byte-level tokenizer does.
"""
from __future__ import annotations

import dataclasses

import pytest

from occ3_sglang.latent_types import LatentError
from occ3_sglang.span_audit import (
    CONTENT, CONTROL_MARKERS, MAX_MARKER_TOKENS, THINKING, UNKNOWN,
    SpanMarkers, classify_accepted, span_from_policy,
)
from occ3_sglang.span_markers import (
    DEFAULT_PROFILE, MARKER_SCHEMA, PROFILES, REASONS, ROLES,
    THINKING_CLOSE, THINKING_OPEN,
    ArgumentDelimiters, ContentDelimiters, MarkerResolutionError, MarkerTable,
    resolve_marker_ids, resolve_marker_table, resolve_stop_ids, vocabulary_bound,
)

BYTE_BASE = 1000            # byte-fallback ids live above every added token
THINK_START, THINK_END = 100, 101
TC_START, TC_END = 102, 103
AK_START, AK_END = 104, 105
AV_START, AV_END = 106, 107
EOS_ID = 108

# One distinct single-token id per GLM spelling (glm4_moe_detector.py:1028-1039).
ADDED = {
    THINKING_OPEN: (THINK_START,),
    THINKING_CLOSE: (THINK_END,),
    "<tool_call>": (TC_START,),
    "</tool_call>": (TC_END,),
    "<arg_key>": (AK_START,),
    "</arg_key>": (AK_END,),
    "<arg_value>": (AV_START,),
    "</arg_value>": (AV_END,),
    "<|endoftext|>": (EOS_ID,),
    "<|assistant|>": (109,),
    "<|user|>": (110,),
}


class FakeTokenizer:
    """Tokenizer-like object: added tokens plus an exact byte fallback."""

    def __init__(self, added=None, *, eos_token="<|endoftext|>", vocab_size=65536,
                 eos_token_id=EOS_ID, eos_token_ids=None,
                 additional_stop_token_ids=None, empty_for=(), decode_suffix="",
                 unk_token_id=None, unk_token=None):
        self.added = {text: tuple(ids) for text, ids in (added or ADDED).items()}
        self.eos_token = eos_token
        self.eos_token_id = eos_token_id
        self.eos_token_ids = eos_token_ids
        self.additional_stop_token_ids = additional_stop_token_ids
        self.vocab_size = vocab_size
        self.empty_for = set(empty_for)
        self.decode_suffix = decode_suffix
        self.unk_token_id = unk_token_id
        self.unk_token = unk_token
        self.calls = []
        self._pieces = {ids[0]: text for text, ids in self.added.items() if len(ids) == 1}

    def encode(self, text, add_special_tokens=True):
        self.calls.append((text, add_special_tokens))
        if add_special_tokens:
            raise AssertionError("marker strings must be encoded without special tokens")
        if text in self.empty_for:
            return []
        if text in self.added:
            return list(self.added[text])
        return [BYTE_BASE + byte for byte in text.encode("utf-8")]

    def decode(self, ids, skip_special_tokens=False):
        parts = []
        for token in ids:
            if token in self._pieces:
                parts.append(self._pieces[token].encode("utf-8"))
            elif BYTE_BASE <= token < BYTE_BASE + 256:
                parts.append(bytes([token - BYTE_BASE]))
            else:
                return "<not-a-marker>"
        try:
            return b"".join(parts).decode("utf-8") + self.decode_suffix
        except UnicodeDecodeError:
            return "<not-a-marker>"


class NoDecodeTokenizer(FakeTokenizer):
    """Same encoder, no ``decode``: attestation must refuse."""

    decode = None


class NoEncodeTokenizer:
    eos_token = "<|endoftext|>"


def table(**kwargs):
    return resolve_marker_table(FakeTokenizer(**kwargs))


def refusal(call, reason, role=None):
    with pytest.raises(MarkerResolutionError) as caught:
        call()
    assert caught.value.reason == reason, (caught.value.reason, caught.value.role)
    assert caught.value.code == "profile"
    assert isinstance(caught.value, LatentError) and isinstance(caught.value, ValueError)
    if role is not None:
        assert caught.value.role == role
    return caught.value


def fixed(value):
    """An encode() replacement that keeps the call signature."""
    return lambda text, add_special_tokens=True: value


# --------------------------------------------------------------------------
# resolution
# --------------------------------------------------------------------------

def test_every_role_resolves_to_one_sequence():
    markers = table()
    assert markers.profile == DEFAULT_PROFILE and markers.schema == MARKER_SCHEMA
    assert tuple(markers.by_role()) == ROLES
    assert markers.complete is True and markers.roundtrip_attested is True
    assert markers.thinking_start_ids == (THINK_START,)
    assert markers.thinking_end_ids == (THINK_END,)
    assert markers.tool_call_start_ids == (TC_START,)
    assert markers.tool_call_end_ids == (TC_END,)
    assert markers.argument_key_start_ids == (AK_START,)
    assert markers.argument_key_end_ids == (AK_END,)
    assert markers.argument_value_start_ids == (AV_START,)
    assert markers.argument_value_end_ids == (AV_END,)
    assert markers.turn_end_ids == (EOS_ID,)
    assert dict(markers.provenance)["turn_end"] == "attribute"
    assert dict(markers.provenance)["thinking_end"] == "literal"


def test_markers_are_encoded_without_special_tokens():
    tokenizer = FakeTokenizer()
    resolve_marker_table(tokenizer)
    assert tokenizer.calls and all(flag is False for _, flag in tokenizer.calls)


def test_content_and_arguments_are_derived_not_invented():
    markers = table()
    content = markers.content
    assert isinstance(content, ContentDelimiters)
    assert content.open_ids == markers.thinking_end_ids
    assert content.close_ids == markers.tool_call_start_ids
    assert content.turn_end_ids == markers.turn_end_ids
    arguments = markers.arguments
    assert isinstance(arguments, ArgumentDelimiters)
    assert arguments.all_ids() == ((AK_START,), (AK_END,), (AV_START,), (AV_END,))


def test_span_markers_bridge_is_the_auditor_contract():
    bridge = table().span_markers()
    assert type(bridge) is SpanMarkers
    assert bridge.thinking_end_ids == (THINK_END,)
    assert bridge.tool_call_begin_ids == (TC_START,)
    assert bridge.tool_call_end_ids == (TC_END,)
    assert bridge.message_end_ids == (EOS_ID,)
    assert bridge.complete is True
    assert bridge.supplied == ("thinking_end_ids", "tool_call_begin_ids",
                               "tool_call_end_ids", "message_end_ids")


def test_table_is_frozen_and_records_are_id_free():
    markers = table()
    with pytest.raises(dataclasses.FrozenInstanceError):
        markers.thinking_end_ids = (1,)
    assert table() == markers and repr(table()) == repr(markers)
    wire = markers.wire()
    assert wire["schema"] == MARKER_SCHEMA and wire["profile"] == DEFAULT_PROFILE
    assert wire["roundtrip_attested"] is True
    assert wire["markers"] == {role: 1 for role in ROLES}
    assert wire["sources"]["turn_end"] == "attribute"
    for text in (repr(markers), str(wire)):
        for fragment in ("<tool_call>", "</tool_call>", "<arg_key>", "<|endoftext|>",
                         "102", "103", "1000"):
            assert fragment not in text


def test_resolve_marker_ids_attests_one_string():
    assert resolve_marker_ids(FakeTokenizer(), "</tool_call>") == (TC_END,)
    assert resolve_marker_ids(FakeTokenizer(), "plain text") == tuple(
        BYTE_BASE + byte for byte in b"plain text")


def test_vocabulary_bound_prefers_vocab_size():
    assert vocabulary_bound(FakeTokenizer(vocab_size=65536)) == 65536
    bare = FakeTokenizer()
    del bare.vocab_size
    assert vocabulary_bound(bare) is None

    class Sized(FakeTokenizer):
        def __len__(self):
            return 4242

    sized = Sized()
    del sized.vocab_size
    assert vocabulary_bound(sized) == 4242
    assert vocabulary_bound(NoEncodeTokenizer()) is None


def test_vocabulary_bound_covers_added_special_tokens():
    """vocab_size counts base embeddings; len() includes added special tokens.

    GLM-5.3 (measured live, window icalib-20261029T032600Z): vocab_size=154820
    while len(tokenizer)=154856, and every GLM-XML marker literal resolves to
    an added token ABOVE vocab_size. The bound must be the larger of the two
    or every marker is refused with reason='vocab' before roundtrip attestation
    (this is exactly what blocked v2 span policies on the live candidate).
    """
    class AddedTokens(FakeTokenizer):
        def __len__(self):
            return 4280

    tok = AddedTokens(vocab_size=4242)
    assert vocabulary_bound(tok) == 4280
    # The marker table resolves when marker ids live in the added-token range.
    assert resolve_marker_table(tok) is not None
    # A larger vocab_size still wins over a smaller len(); the max never lowers
    # an honest count.
    big = FakeTokenizer(vocab_size=65536)
    big.__class__ = big.__class__  # no len override: falls back to vocab_size
    assert vocabulary_bound(big) == 65536


def test_profiles_are_the_reviewed_set():
    assert PROFILES == (DEFAULT_PROFILE,)
    assert "glm45" in DEFAULT_PROFILE and "glm47" in DEFAULT_PROFILE
    refusal(lambda: resolve_marker_table(FakeTokenizer(), profile="glm99"), "profile")
    refusal(lambda: resolve_marker_table(FakeTokenizer(), profile=None), "profile")
    refusal(lambda: resolve_marker_table(FakeTokenizer(), allow_absent=["thinking_start"]),
            "profile")


# --------------------------------------------------------------------------
# ambiguity refusal
# --------------------------------------------------------------------------

def test_disagreeing_corroborating_literal_is_ambiguous():
    refusal(lambda: resolve_marker_table(FakeTokenizer(eos_token="<|user|>")),
            "ambiguous", "turn_end")


def test_shared_sequence_is_ambiguous_without_roundtrip():
    # With round-trip attestation on, a hand-made collision is caught earlier as
    # "unattested": decode() can only return one text per id sequence.  The
    # structural check is the fallback for require_roundtrip=False.
    added = dict(ADDED, **{"</tool_call>": (TC_START,)})
    refusal(lambda: resolve_marker_table(FakeTokenizer(added)), "unattested")
    refusal(lambda: resolve_marker_table(FakeTokenizer(added), require_roundtrip=False),
            "ambiguous", "tool_call_end")


def test_proper_prefix_across_roles_is_refused_without_roundtrip():
    added = dict(ADDED, **{"</tool_call>": (TC_START, 999)})
    refusal(lambda: resolve_marker_table(FakeTokenizer(added), require_roundtrip=False),
            "prefix", "tool_call_start")


def test_multi_token_marker_is_accepted_when_it_roundtrips():
    # A marker need not be one added token.  A byte-level tokenizer spells a
    # special string in several pieces, and decode() still reconstructs it, so
    # the sequence is exact and the table stays complete.
    added = {text: ids for text, ids in ADDED.items() if text != "<arg_value>"}
    markers = resolve_marker_table(FakeTokenizer(added))
    assert markers.argument_value_start_ids == body("<arg_value>")
    assert len(markers.argument_value_start_ids) > 1
    assert markers.complete is True


# --------------------------------------------------------------------------
# missing-marker refusal
# --------------------------------------------------------------------------

def test_no_tokenizer_is_refused():
    refusal(lambda: resolve_marker_table(NoEncodeTokenizer()), "no_tokenizer")


def test_unencodable_required_marker_is_missing():
    refusal(lambda: resolve_marker_table(FakeTokenizer(empty_for=("</tool_call>",))),
            "missing", "tool_call_end")


def test_turn_end_without_a_tokenizer_declaration_is_missing():
    refusal(lambda: resolve_marker_table(FakeTokenizer(eos_token=None)), "missing", "turn_end")
    refusal(lambda: resolve_marker_table(FakeTokenizer(eos_token="")), "missing", "turn_end")


def test_turn_end_literal_alone_cannot_attest():
    class LiteralOnly(FakeTokenizer):
        def encode(self, text, add_special_tokens=True):
            if text == "<|endoftext|>":
                return [EOS_ID]
            return super().encode(text, add_special_tokens=add_special_tokens)

    refusal(lambda: resolve_marker_table(LiteralOnly(eos_token=None)), "missing", "turn_end")


def test_narrow_vocabulary_refuses_the_marker():
    refusal(lambda: resolve_marker_table(FakeTokenizer(vocab_size=64)), "vocab")


# --------------------------------------------------------------------------
# attestation refusal
# --------------------------------------------------------------------------

def test_normalised_decode_is_not_attested():
    refusal(lambda: resolve_marker_table(FakeTokenizer(decode_suffix="\x00")), "unattested")


def test_tokenizer_without_decode_is_not_attested():
    refusal(lambda: resolve_marker_table(NoDecodeTokenizer()), "unattested")


def test_roundtrip_can_be_disabled_and_unknown_substitution_still_refuses():
    markers = resolve_marker_table(FakeTokenizer(), require_roundtrip=False)
    assert markers.roundtrip_attested is False and markers.complete is True
    unk = FakeTokenizer(added={}, unk_token_id=7, unk_token="<unk>")
    unk.encode = fixed([7])
    refusal(lambda: resolve_marker_table(unk, require_roundtrip=False), "unattested")


@pytest.mark.parametrize("value,reason", [
    (None, "shape"),
    ("101", "shape"),
    ([THINK_END, True], "shape"),
    ([THINK_END, -1], "shape"),
    ([THINK_END, 1.5], "shape"),
    (list(range(MAX_MARKER_TOKENS + 1)), "bounds"),
    (THINK_END, "shape"),
])
def test_bad_encode_shapes_fail_closed(value, reason):
    tokenizer = FakeTokenizer()
    tokenizer.encode = fixed(value)
    refusal(lambda: resolve_marker_table(tokenizer), reason)


def test_empty_encode_result_is_missing():
    tokenizer = FakeTokenizer()
    tokenizer.encode = fixed([])
    refusal(lambda: resolve_marker_table(tokenizer), "missing")


def test_array_like_encode_results_are_accepted():
    class Array:
        def __init__(self, values):
            self.values = values

        def tolist(self):
            return self.values

    tokenizer = FakeTokenizer()
    original = FakeTokenizer().encode
    tokenizer.encode = lambda text, add_special_tokens=True: Array(
        original(text, add_special_tokens=add_special_tokens))
    assert resolve_marker_table(tokenizer).thinking_end_ids == (THINK_END,)


def test_exception_carries_no_marker_text_or_ids():
    error = refusal(lambda: resolve_marker_table(FakeTokenizer(empty_for=("</tool_call>",))),
                    "missing", "tool_call_end")
    assert "</tool_call>" not in repr(error) and "103" not in repr(error)
    assert str(error) == "profile"


def test_reasons_are_a_fixed_allowlist():
    assert "missing" in REASONS and "prefix" in REASONS
    hidden = MarkerResolutionError("not-a-reason", "not-a-role")
    assert hidden.reason == "profile" and hidden.role == ""


# --------------------------------------------------------------------------
# declared absence
# --------------------------------------------------------------------------

def test_allowlisted_absent_thinking_start_is_recorded_not_guessed():
    tokenizer = FakeTokenizer(empty_for=(THINKING_OPEN,))
    refusal(lambda: resolve_marker_table(tokenizer), "missing", "thinking_start")
    markers = resolve_marker_table(tokenizer, allow_absent=("thinking_start",))
    assert markers.thinking_start_ids is None
    assert dict(markers.provenance)["thinking_start"] == "absent"
    assert markers.complete is False
    assert markers.wire()["markers"]["thinking_start"] is None
    assert markers.span_markers().thinking_end_ids == (THINK_END,)


def test_only_absentable_roles_may_be_absent():
    refusal(lambda: resolve_marker_table(FakeTokenizer(), allow_absent=("thinking_end",)),
            "profile")
    refusal(lambda: resolve_marker_table(FakeTokenizer(), allow_absent=("nope",)), "profile")


# --------------------------------------------------------------------------
# stop-id set
# --------------------------------------------------------------------------

def test_stop_ids_union_every_declared_source():
    tokenizer = FakeTokenizer(eos_token_ids=[EOS_ID, 110], additional_stop_token_ids=(111,))
    assert resolve_stop_ids(tokenizer) == frozenset({EOS_ID, 110, 111})
    assert resolve_stop_ids(FakeTokenizer(), extra=(112,)) == frozenset({EOS_ID, 112})


def test_stop_ids_refuse_an_empty_or_out_of_range_set():
    refusal(lambda: resolve_stop_ids(FakeTokenizer(eos_token_id=None)), "missing", "turn_end")
    refusal(lambda: resolve_stop_ids(FakeTokenizer(vocab_size=64)), "vocab", "turn_end")


# --------------------------------------------------------------------------
# integration with span_audit.classify_accepted
# --------------------------------------------------------------------------

def body(text):
    return tuple(BYTE_BASE + byte for byte in text.encode("utf-8"))


def positions(ids):
    return list(range(len(ids)))


def test_integration_content_then_tool_call():
    markers = table()
    content = body("Paris is the capital.")
    ids = (markers.thinking_end_ids + content
           + markers.tool_call_start_ids + body("occ3_handoff")
           + markers.argument_key_start_ids + body("value")
           + markers.argument_key_end_ids + markers.argument_value_start_ids
           + body("42") + markers.argument_value_end_ids
           + markers.tool_call_end_ids + markers.turn_end_ids)
    audit = classify_accepted(ids, positions(ids), markers.span_markers())
    assert audit.contains_final_content is True and audit.contains_tool_call is True
    assert audit.tool_call_complete is True and audit.message_truncated is False
    assert audit.counts[CONTENT] == len(content)
    selection = span_from_policy(audit, "final_content")
    assert selection.bounds == (1, 1 + len(content))
    assert selection.contiguous is True
    assert span_from_policy(audit, "final_content_no_tool") is None
    tail = span_from_policy(audit, "last_accepted")
    assert tail.bounds == (0, len(ids))
    assert tail.end_index - tail.start_index > selection.length


def test_integration_tool_only_turn_has_no_content():
    markers = table()
    ids = (markers.tool_call_start_ids + body("occ3_handoff")
           + markers.argument_key_start_ids + body("value")
           + markers.argument_key_end_ids + markers.argument_value_start_ids
           + body("42") + markers.argument_value_end_ids
           + markers.tool_call_end_ids + markers.turn_end_ids)
    audit = classify_accepted(ids, positions(ids), markers.span_markers())
    # SpanMarkers carries four markers, so only tool_call_start, tool_call_end and
    # the turn end become CONTROL rows; the four argument delimiters are not
    # expressible there and fall into TOOL_CALL / TOOL_ARGUMENTS by region.
    assert audit.counts[CONTENT] == 0 and audit.counts[CONTROL_MARKERS] == 3
    assert audit.counts[UNKNOWN] == 0
    assert audit.contains_final_content is False
    for policy in ("final_content", "final_content_no_tool", "final_content_last64"):
        assert span_from_policy(audit, policy) is None
    assert span_from_policy(audit, "last_accepted").bounds == (0, len(ids))


def test_integration_reasoning_terminated_by_tool_call_stays_unclassified():
    markers = table()
    # Glm45Detector lets <tool_call> end reasoning with no  response
    # (reasoning_parser.py:936, 165-177).  The auditor must not call that prose.
    reasoning = body("thinking about it")
    ids = (markers.thinking_start_ids + reasoning
           + markers.tool_call_start_ids + body("occ3_handoff")
           + markers.tool_call_end_ids + markers.turn_end_ids)
    audit = classify_accepted(ids, positions(ids), markers.span_markers())
    assert audit.counts[THINKING] == 0
    assert audit.counts[UNKNOWN] >= len(reasoning)
    assert audit.contains_final_content is False
    assert span_from_policy(audit, "final_content") is None


def test_integration_missing_turn_end_marker_is_reported_not_raised():
    markers = table()
    content = body("short answer")
    ids = markers.thinking_end_ids + content
    audit = classify_accepted(ids, positions(ids), markers.span_markers())
    assert audit.contains_final_content is True
    assert ("message_end_ids", 0) in audit.marker_occurrences
    assert span_from_policy(audit, "final_content").bounds == (1, 1 + len(content))


# --------------------------------------------------------------------------
# purity
# --------------------------------------------------------------------------

def test_module_is_pure_python_and_never_imports_sglang():
    module = __import__("occ3_sglang.span_markers", fromlist=["__file__"])
    source = open(module.__file__, "r", encoding="utf-8").read()
    for forbidden in ("import sglang", "from sglang", "import torch", "from torch",
                      "open(", "requests.", "socket"):
        assert forbidden not in source
    assert module.MARKER_SCHEMA == "occ3.span.markers.v1"
    assert module.DEFAULT_PROFILE == DEFAULT_PROFILE
    assert issubclass(MarkerTable, object)


# --------------------------------------------------------------------------
# GLM-5.3 shapes (HF zai-org/GLM-5.3: tokenizer.vocab_size=154820,
# len(tokenizer)=154856, config.vocab_size=154880 padded embedding rows,
# generation_config.json eos_token_id=[154820, 154827, 154829]).
# --------------------------------------------------------------------------

from types import SimpleNamespace as NS  # noqa: E402

GLM53_BASE = 154820             # tokenizer.vocab_size; <|endoftext|> is the first added id
GLM53_LEN = 154856              # len(tokenizer): base plus 36 added specials
GLM53_PADDED = 154880           # config.vocab_size: embedding rows, never token ids
GLM53_EOS = GLM53_BASE          # <|endoftext|> (eos and pad)
GLM53_USER = GLM53_BASE + 7     # <|user|>: ends an answer turn
GLM53_ASSISTANT = GLM53_BASE + 8
GLM53_OBSERVATION = GLM53_BASE + 9  # <|observation|>: ends a tool-calling turn
GLM53_THINK = GLM53_BASE + 21   # <think> (154841); the XML markers follow in order
GLM53_ADDED = {
    "<|endoftext|>": (GLM53_EOS,), "[MASK]": (GLM53_BASE + 1,), "[gMASK]": (GLM53_BASE + 2,),
    "<sop>": (GLM53_BASE + 3,), "<eop>": (GLM53_BASE + 4,), "<|system|>": (GLM53_BASE + 5,),
    "<|user|>": (GLM53_USER,), "<|assistant|>": (GLM53_ASSISTANT,),
    "<|observation|>": (GLM53_OBSERVATION,),
    THINKING_OPEN: (GLM53_THINK,), THINKING_CLOSE: (GLM53_THINK + 1,),
    "<tool_call>": (GLM53_THINK + 2,), "</tool_call>": (GLM53_THINK + 3,),
    "<arg_key>": (GLM53_THINK + 4,), "</arg_key>": (GLM53_THINK + 5,),
    "<arg_value>": (GLM53_THINK + 6,), "</arg_value>": (GLM53_THINK + 7,),
}
GLM53_GENERATION_EOS = (GLM53_EOS, GLM53_USER, GLM53_OBSERVATION)


class Glm53Tokenizer(FakeTokenizer):
    """GLM-5.3-shaped tokenizer: base vocab 154820, added specials to 154856."""

    def __init__(self, **kwargs):
        kwargs.setdefault("added", GLM53_ADDED)
        kwargs.setdefault("vocab_size", GLM53_BASE)
        kwargs.setdefault("eos_token_id", GLM53_EOS)
        super().__init__(**kwargs)

    def __len__(self):
        return GLM53_LEN


def test_glm53_marker_table_resolves_every_role_above_the_base_vocab():
    tok = Glm53Tokenizer()
    assert vocabulary_bound(tok) == GLM53_LEN
    markers = resolve_marker_table(tok)
    assert markers.complete and markers.roundtrip_attested
    for role, ids in markers.by_role().items():
        # Every GLM-XML marker and the eos itself sit at or above the base
        # vocab_size (the pre-#212 bound refused all nine with reason='vocab')
        # and strictly below len(tokenizer); padded rows are never token ids.
        assert len(ids) == 1 and GLM53_BASE <= ids[0] < GLM53_LEN < GLM53_PADDED, role
    assert markers.turn_end_ids == (GLM53_EOS,) and GLM53_EOS >= tok.vocab_size
    assert markers.thinking_start_ids == (GLM53_THINK,)
    assert markers.thinking_end_ids == (GLM53_THINK + 1,)
    assert markers.tool_call_start_ids == (GLM53_THINK + 2,)


def test_glm53_turn_stop_set_covers_the_role_tokens_a_turn_really_ends_on():
    """Red before the stop-set fix: the stop set was {<|endoftext|>} only.

    The GLM-5.3 template has no end-of-turn token; the model ends an answer
    on <|user|> and a tool-calling turn on <|observation|>, and SGLang stops
    there through the generation config's eos set (req.eos_token_ids).
    """
    tok = Glm53Tokenizer()
    assert resolve_stop_ids(tok) == frozenset((GLM53_EOS,))
    config = NS(hf_eos_token_id=set(GLM53_GENERATION_EOS))
    assert resolve_turn_stop_ids(tok, config) == frozenset(GLM53_GENERATION_EOS)
    # Without the model config the profile's attested role spellings still cover it.
    assert resolve_turn_stop_ids(tok) == frozenset(GLM53_GENERATION_EOS)
    # int and list forms of hf_eos_token_id are accepted; booleans are ignored.
    assert resolve_turn_stop_ids(tok, NS(hf_eos_token_id=GLM53_USER)) == frozenset(GLM53_GENERATION_EOS)
    assert resolve_turn_stop_ids(tok, NS(hf_eos_token_id=[GLM53_EOS, True])) == frozenset(GLM53_GENERATION_EOS)
    # The role tokens are encoded exactly like the markers: no special-token prefix.
    assert all(flag is False for _, flag in tok.calls)


def test_turn_stop_set_skips_role_spellings_the_tokenizer_lacks():
    # test_span_markers.FakeTokenizer knows <|user|> (110) but spells
    # <|observation|> through byte fallback: not one id, so not a stop id.
    assert resolve_turn_stop_ids(FakeTokenizer()) == frozenset((EOS_ID, 110))
    bare = FakeTokenizer(added={k: v for k, v in ADDED.items() if k != "<|user|>"})
    assert resolve_turn_stop_ids(bare) == frozenset((EOS_ID,))
    # A generation-config id outside the tokenizer id space (a padded
    # embedding row) is refused, never trusted.
    refusal(lambda: resolve_turn_stop_ids(Glm53Tokenizer(), NS(hf_eos_token_id={GLM53_PADDED - 1})),
            "vocab", "turn_end")
    refusal(lambda: resolve_turn_stop_ids(Glm53Tokenizer(), profile="invented.v9"), "profile")
