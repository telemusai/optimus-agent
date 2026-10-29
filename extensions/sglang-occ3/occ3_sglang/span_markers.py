"""Tokenizer-attested marker ids for a GLM reasoning/tool-call source turn.

Why this module exists
----------------------
``span_audit.classify_accepted`` classifies accepted rows by exact token-id
markers, but it trusts whatever ids it is handed.  Nothing in this extension can
produce those ids: the extension holds no tokenizer, and the ids must come from
the tokenizer the served model actually uses.  This module turns a tokenizer-like
object plus a versioned marker profile into one frozen, auditable table, and
refuses instead of guessing.

Attestation contract
--------------------
Every role in the profile must resolve to **exactly one** id sequence, and

* ``tokenizer.encode(text, add_special_tokens=False)`` must return a non-empty,
  bounded sequence of non-negative plain ints, inside the vocabulary when a
  vocabulary bound is knowable;
* ``tokenizer.decode(ids)`` must return ``text`` exactly, so an unknown-token
  substitution, a byte-fallback rewrite or a normalisation cannot pass as a
  marker;
* two roles may not share a sequence, and no role's sequence may be a proper
  prefix of another role's sequence: ``span_audit`` matches the first
  non-overlapping occurrence of each marker and then requires one strict nesting
  order, so a prefix pair makes the result depend on match order;
* a role may be declared absent only when the profile marks it absentable *and*
  the caller allowlists that role by name;
* a role whose profile names a tokenizer attribute (the turn-end role names
  ``eos_token``) is attested only by that attribute; a corroborating literal
  must agree with it and can never replace it, because byte-fallback text can
  encode cleanly while the model never emits it.

Any other outcome raises :class:`MarkerResolutionError` with a bounded ``reason``
and ``role``.  Marker strings, decoded text and token ids never enter the
exception, ``repr()`` or :meth:`MarkerTable.wire`.

Marker provenance
-----------------
The marker strings are not invented here.  They are the strings the pinned
SGLang parsers themselves match, per profile.

``glm45_glm47_xml_kv.v1`` - GLM-4.5 / 4.6 / 4.7 / 5.x served with
``--reasoning-parser glm45 --tool-call-parser glm47``:

===================== =================== ======================================
role                  string              pinned source
===================== =================== ======================================
thinking_start        ``" thinking"``       glm4_moe_detector.py:1028,
                                          reasoning_parser.py:931
thinking_end          ``"</think>"``      glm4_moe_detector.py:1029,
                                          reasoning_parser.py:932
tool_call_start       ``"<tool_call>"``    glm4_moe_detector.py:174,
                                          glm47_moe_detector.py:311
tool_call_end         ``"</tool_call>"``   glm4_moe_detector.py:175,
                                          glm47_moe_detector.py:312
argument_key_start    ``"<arg_key>"``      glm4_moe_detector.py:1032,
                                          glm47_moe_detector.py:315
argument_key_end      ``"</arg_key>"``     glm4_moe_detector.py:1033,
                                          glm47_moe_detector.py:318
argument_value_start  ``"<arg_value>"``    glm4_moe_detector.py:1034,
                                          glm47_moe_detector.py:318
argument_value_end    ``"</arg_value>"``   glm4_moe_detector.py:1035,
                                          glm47_moe_detector.py:318
turn_end              the tokenizer's own schedule_batch.py:1783-1803
                      ``eos_token``
===================== =================== ======================================

The GLM template has no ``content`` tag: ``assistant_turn ::= thinking_block
text_block tool_call_blocks`` (glm4_moe_detector.py:1057), so a content row is
*derived* as "after ``thinking_end``, before ``tool_call_start`` or the turn
end".  :attr:`MarkerTable.content` exposes that derivation instead of pretending
a tag exists.  Reasoning can also end at ``tool_call_start`` with no
``thinking_end`` at all, because ``Glm45Detector`` passes
``tool_start_token="<tool_call>"`` and interrupts reasoning there
(reasoning_parser.py:936, 165-177, 252-267).

What this module does not do
----------------------------
It does not import SGLang, does not read files, does not log, does not wire
itself into any capture path and does not select a span.  It resolves and attests
markers; selection stays in ``span_audit``.
"""
from __future__ import annotations

import operator
from dataclasses import dataclass
from typing import Dict, Tuple

from .attachment_contract import MARKER_SCHEMA_V1
from .latent_types import LatentError
from .span_audit import MAX_MARKER_ID, MAX_MARKER_TOKENS, SpanMarkers

__all__ = [
    "DEFAULT_PROFILE", "MARKER_SCHEMA", "PROFILES", "REASONS", "ROLES",
    "THINKING_CLOSE", "THINKING_OPEN",
    "MarkerResolutionError", "ArgumentDelimiters", "ContentDelimiters",
    "MarkerTable", "resolve_marker_ids", "resolve_marker_table",
    "resolve_stop_ids", "vocabulary_bound",
]

#: Version of the resolved-table contract.  A receipt that names a marker
#: profile must name both this schema and the profile string below.
#: Single-sourced from the binding contract module so the v2 receipt block
#: and the marker table can never drift apart.
MARKER_SCHEMA = MARKER_SCHEMA_V1

#: Role order is fixed so every tuple and dict in this module is comparable.
ROLES = (
    "thinking_start", "thinking_end",
    "tool_call_start", "tool_call_end",
    "argument_key_start", "argument_key_end",
    "argument_value_start", "argument_value_end",
    "turn_end",
)

#: Supported marker profiles.  Adding one is a review event, not a config knob.
DEFAULT_PROFILE = "glm45_glm47_xml_kv.v1"
PROFILES = (DEFAULT_PROFILE,)

#: The two thinking delimiters, spelled with escapes so that neither a source
#: editor nor a downstream sanitiser can rewrite the angle brackets.  These are
#: the exact strings of glm4_moe_detector.py:1028-1029 and
#: reasoning_parser.py:931-932: U+003C "think" U+003E and U+003C "/" "think"
#: U+003E.
THINKING_OPEN = "\x3cthink\x3e"
THINKING_CLOSE = "\x3c/think\x3e"

#: Fixed refusal reasons.  Free-form text would leak marker strings or ids.
REASONS = (
    "profile", "no_tokenizer", "encode_failed", "shape", "bounds", "vocab",
    "missing", "ambiguous", "prefix", "unattested",
)


class MarkerResolutionError(LatentError, ValueError):
    """Fail-closed refusal to build a marker table.

    A ``LatentError`` so worker seams handle it like every other decline, and a
    ``ValueError`` so an ordinary caller can catch it.  ``code`` is the shared
    ``LatentError`` profile code; ``reason`` and ``role`` are fixed identifiers
    and never carry a marker string or a token id.
    """

    def __init__(self, reason: str = "profile", role: str = ""):
        self.reason = reason if reason in REASONS else "profile"
        self.role = role if role in ROLES else ""
        super().__init__("profile")

    def __repr__(self) -> str:
        return "MarkerResolutionError(code=%r, reason=%r, role=%r)" % (
            self.code, self.reason, self.role)


@dataclass(frozen=True)
class _Spec:
    """One role's candidate spellings for one profile."""

    role: str
    literals: Tuple[str, ...] = ()
    attributes: Tuple[str, ...] = ()
    absentable: bool = False


_GLM_XML_KV = (
    _Spec("thinking_start", literals=(THINKING_OPEN,), absentable=True),
    _Spec("thinking_end", literals=(THINKING_CLOSE,)),
    _Spec("tool_call_start", literals=("<tool_call>",)),
    _Spec("tool_call_end", literals=("</tool_call>",)),
    _Spec("argument_key_start", literals=("<arg_key>",)),
    _Spec("argument_key_end", literals=("</arg_key>",)),
    _Spec("argument_value_start", literals=("<arg_value>",)),
    _Spec("argument_value_end", literals=("</arg_value>",)),
    # The turn-end marker is whatever this tokenizer declares as EOS; the
    # literal only corroborates it and must agree.  A parser that can stop on a
    # second id is not attestable by one sequence - see resolve_stop_ids.
    _Spec("turn_end", literals=("<|endoftext|>",), attributes=("eos_token",)),
)

_PROFILES = {DEFAULT_PROFILE: _GLM_XML_KV}


def _error(reason: str, role: str = "") -> MarkerResolutionError:
    return MarkerResolutionError(reason, role)


def vocabulary_bound(tokenizer) -> int | None:
    """Best-effort vocabulary size, or None when the object does not say.

    ``vocab_size`` wins over ``len(tokenizer)``.  A non-positive or non-int
    answer is ignored rather than trusted: a bound that is too small would
    reject every marker, and one that is invented would accept an id the model
    cannot emit.
    """
    size = getattr(tokenizer, "vocab_size", None)
    if type(size) is int and size > 0:
        return size
    try:
        length = len(tokenizer)
    except Exception:
        return None
    if type(length) is int and length > 0:
        return length
    return None


def _plain_ids(value, role: str) -> Tuple[int, ...]:
    """Normalise one ``encode()`` result to a bounded tuple of plain ints."""
    if value is None or isinstance(value, (str, bytes, bytearray)):
        raise _error("shape", role)
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        try:
            value = tolist()
        except Exception:
            raise _error("shape", role) from None
    try:
        items = tuple(value)
    except TypeError:
        raise _error("shape", role) from None
    if not items:
        raise _error("missing", role)
    if len(items) > MAX_MARKER_TOKENS:
        raise _error("bounds", role)
    ids = []
    for item in items:
        if isinstance(item, bool) or isinstance(item, (str, bytes, bytearray)):
            raise _error("shape", role)
        try:
            number = operator.index(item)
        except TypeError:
            raise _error("shape", role) from None
        if number < 0 or number > MAX_MARKER_ID:
            raise _error("shape", role)
        ids.append(int(number))
    return tuple(ids)


def _decode(tokenizer, ids: Tuple[int, ...], role: str) -> str:
    """Decode one attested sequence, refusing a silent fallback."""
    decode = getattr(tokenizer, "decode", None)
    if not callable(decode):
        raise _error("unattested", role)
    try:
        text = decode(list(ids), skip_special_tokens=False)
    except TypeError:
        try:
            text = decode(list(ids))
        except Exception:
            raise _error("unattested", role) from None
    except Exception:
        raise _error("unattested", role) from None
    if type(text) is not str:
        raise _error("unattested", role)
    return text


def _encode(tokenizer, text: str, role: str) -> Tuple[int, ...]:
    """Encode one candidate string and return its exact id sequence."""
    if type(text) is not str or not text:
        raise _error("shape", role)
    encode = getattr(tokenizer, "encode", None)
    if not callable(encode):
        raise _error("no_tokenizer", role)
    try:
        # Same call shape the pinned source uses for marker strings:
        # reasoning_parser.py:117-118, serving_chat.py:1443-1446,
        # reasoner_grammar_backend.py:274-276.
        raw = encode(text, add_special_tokens=False)
    except TypeError:
        raise _error("no_tokenizer", role) from None
    except Exception:
        raise _error("encode_failed", role) from None
    return _plain_ids(raw, role)


def _candidate_ids(tokenizer, text: str, role: str, bound, *,
                   require_roundtrip: bool) -> Tuple[int, ...]:
    """Encode plus vocabulary and round-trip attestation for one string."""
    ids = _encode(tokenizer, text, role)
    if bound is not None and any(token >= bound for token in ids):
        raise _error("vocab", role)
    if require_roundtrip:
        if _decode(tokenizer, ids, role) != text:
            raise _error("unattested", role)
    else:
        unk = getattr(tokenizer, "unk_token_id", None)
        unk_token = getattr(tokenizer, "unk_token", None)
        if type(unk) is int and ids == (int(unk),) and text != unk_token:
            raise _error("unattested", role)
    return ids


def resolve_marker_ids(tokenizer, text: str, *, role: str = "",
                       require_roundtrip: bool = True) -> Tuple[int, ...]:
    """Resolve one marker string to exactly one attested id sequence.

    Public so a caller or a test can attest a single marker without building a
    whole table.  ``require_roundtrip`` is the only escape from decode
    attestation: a caller that disables it accepts unknown-token substitution in
    exchange for a tokenizer without ``decode``.
    """
    return _candidate_ids(tokenizer, text, role, vocabulary_bound(tokenizer),
                          require_roundtrip=require_roundtrip)


def _resolve_role(tokenizer, spec: _Spec, bound, allow_absent, require_roundtrip):
    """Return ``(ids | None, source)`` for one role, or raise."""
    authoritative = []
    for name in spec.attributes:
        value = getattr(tokenizer, name, None)
        if type(value) is str and value:
            authoritative.append(value)
    resolved = []
    for text in authoritative + list(spec.literals):
        source = "attribute" if text in authoritative else "literal"
        try:
            ids = _candidate_ids(tokenizer, text, spec.role, bound,
                                 require_roundtrip=require_roundtrip)
        except MarkerResolutionError as error:
            if error.reason == "missing":
                continue  # this spelling does not exist for this tokenizer
            raise
        resolved.append((ids, source))
    if not resolved:
        if spec.absentable and spec.role in allow_absent:
            return None, "absent"
        raise _error("missing", spec.role)
    if spec.attributes and not any(source == "attribute" for _, source in resolved):
        # A corroborating literal without the tokenizer's own declaration is not
        # an attestation: byte-fallback text encodes cleanly and the model may
        # never emit it.
        raise _error("missing", spec.role)
    sequences = {ids for ids, _ in resolved}
    if len(sequences) > 1:
        raise _error("ambiguous", spec.role)
    ids = sequences.pop()
    source = "attribute" if any(seq == ids and kind == "attribute"
                                for seq, kind in resolved) else "literal"
    return ids, source


def _check_table(table: "MarkerTable") -> None:
    """Refuse equality and proper-prefix relations across resolved roles."""
    present = [(role, ids) for role, ids in table.by_role().items() if ids is not None]
    for index, (role_a, ids_a) in enumerate(present):
        for role_b, ids_b in present[index + 1:]:
            if ids_a == ids_b:
                raise _error("ambiguous", role_b)
            if len(ids_a) < len(ids_b) and ids_b[:len(ids_a)] == ids_a:
                raise _error("prefix", role_a)
            if len(ids_b) < len(ids_a) and ids_a[:len(ids_b)] == ids_b:
                raise _error("prefix", role_b)


@dataclass(frozen=True, repr=False)
class ContentDelimiters:
    """The content region, derived from markers that really exist.

    ``open_ids`` is the marker whose end opens content, ``close_ids`` the marker
    that ends it, and ``turn_end_ids`` the marker that ends it when no tool call
    follows.
    """

    open_ids: Tuple[int, ...]
    close_ids: Tuple[int, ...]
    turn_end_ids: Tuple[int, ...]


@dataclass(frozen=True, repr=False)
class ArgumentDelimiters:
    """The four GLM XML key/value delimiters that frame tool-call arguments."""

    key_start_ids: Tuple[int, ...]
    key_end_ids: Tuple[int, ...]
    value_start_ids: Tuple[int, ...]
    value_end_ids: Tuple[int, ...]

    def all_ids(self) -> Tuple[Tuple[int, ...], ...]:
        return (self.key_start_ids, self.key_end_ids,
                self.value_start_ids, self.value_end_ids)


@dataclass(frozen=True, repr=False)
class MarkerTable:
    """Frozen, attested marker ids for one tokenizer and one profile.

    Built only by :func:`resolve_marker_table`.  Every marker field is a tuple of
    plain ints, or None for a role the profile marks absentable that the caller
    explicitly allowlisted as absent.
    """

    profile: str
    schema: str
    roundtrip_attested: bool
    provenance: Tuple[Tuple[str, str], ...]
    thinking_start_ids: Tuple[int, ...] | None
    thinking_end_ids: Tuple[int, ...]
    tool_call_start_ids: Tuple[int, ...]
    tool_call_end_ids: Tuple[int, ...]
    argument_key_start_ids: Tuple[int, ...]
    argument_key_end_ids: Tuple[int, ...]
    argument_value_start_ids: Tuple[int, ...]
    argument_value_end_ids: Tuple[int, ...]
    turn_end_ids: Tuple[int, ...]

    def by_role(self) -> Dict[str, Tuple[int, ...] | None]:
        """Role -> ids, in fixed :data:`ROLES` order."""
        return {role: getattr(self, role + "_ids") for role in ROLES}

    @property
    def content(self) -> ContentDelimiters:
        """Derived content region; the GLM template has no content tag."""
        return ContentDelimiters(open_ids=self.thinking_end_ids,
                                 close_ids=self.tool_call_start_ids,
                                 turn_end_ids=self.turn_end_ids)

    @property
    def arguments(self) -> ArgumentDelimiters:
        return ArgumentDelimiters(self.argument_key_start_ids, self.argument_key_end_ids,
                                  self.argument_value_start_ids, self.argument_value_end_ids)

    @property
    def complete(self) -> bool:
        """True when every role is present, not only the required ones."""
        return all(ids is not None for ids in self.by_role().values())

    def span_markers(self) -> SpanMarkers:
        """The four markers ``span_audit`` understands, in its field names."""
        return SpanMarkers(thinking_end_ids=self.thinking_end_ids,
                           tool_call_begin_ids=self.tool_call_start_ids,
                           tool_call_end_ids=self.tool_call_end_ids,
                           message_end_ids=self.turn_end_ids)

    def wire(self) -> dict:
        """Bounded descriptor: profile, shapes and sources only, never ids."""
        return {
            "schema": self.schema,
            "profile": self.profile,
            "roundtrip_attested": self.roundtrip_attested,
            "markers": {role: (None if ids is None else len(ids))
                        for role, ids in self.by_role().items()},
            "sources": dict(self.provenance),
        }

    def __repr__(self) -> str:
        shapes = ", ".join("%s=%s" % (role, "None" if ids is None else len(ids))
                           for role, ids in self.by_role().items())
        return "MarkerTable(%s, roundtrip=%s, %s)" % (
            self.profile, self.roundtrip_attested, shapes)


def resolve_marker_table(tokenizer, *, profile: str = DEFAULT_PROFILE,
                         require_roundtrip: bool = True,
                         allow_absent: Tuple[str, ...] = ()) -> MarkerTable:
    """Build the frozen marker table for ``tokenizer``, or refuse.

    ``allow_absent`` may name only roles the profile marks absentable; any other
    name is a profile error, never a silent gap.
    """
    if type(profile) is not str or profile not in _PROFILES:
        raise _error("profile")
    if type(allow_absent) is not tuple:
        raise _error("profile")
    specs = _PROFILES[profile]
    absentable = {spec.role for spec in specs if spec.absentable}
    for role in allow_absent:
        if type(role) is not str or role not in ROLES or role not in absentable:
            raise _error("profile")
    bound = vocabulary_bound(tokenizer)
    values = {}
    provenance = []
    for spec in specs:
        ids, source = _resolve_role(tokenizer, spec, bound, allow_absent,
                                    require_roundtrip)
        values[spec.role + "_ids"] = ids
        provenance.append((spec.role, source))
    table = MarkerTable(profile=profile, schema=MARKER_SCHEMA,
                        roundtrip_attested=bool(require_roundtrip),
                        provenance=tuple(provenance), **values)
    _check_table(table)
    return table


def resolve_stop_ids(tokenizer, extra=()) -> frozenset:
    """The bounded set of token ids that can end this model's turn.

    Not a marker.  Pinned SGLang ends a turn on any id in ``stop_token_ids``,
    ``eos_token_ids``, ``eos_token_id`` or ``additional_stop_token_ids``
    (schedule_batch.py:1783-1803), so one marker sequence cannot attest a turn
    end by itself.  A policy that must prove "the window ends at the message
    boundary" intersects this set with the last accepted row.
    """
    ids = set()

    def add(value):
        if isinstance(value, bool):
            return
        if type(value) is int:
            if 0 <= value <= MAX_MARKER_ID:
                ids.add(int(value))
            return
        if isinstance(value, (list, tuple, set, frozenset)):
            for item in value:
                add(item)

    add(getattr(tokenizer, "eos_token_id", None))
    add(getattr(tokenizer, "eos_token_ids", None))
    add(getattr(tokenizer, "additional_stop_token_ids", None))
    add(extra)
    if not ids:
        raise _error("missing", "turn_end")
    bound = vocabulary_bound(tokenizer)
    if bound is not None and any(token >= bound for token in ids):
        raise _error("vocab", "turn_end")
    if len(ids) > MAX_MARKER_TOKENS:
        raise _error("bounds", "turn_end")
    return frozenset(ids)
