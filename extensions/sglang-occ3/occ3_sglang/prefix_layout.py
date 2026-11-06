
"""Receiver continuous-prefix placement: marker resolution and span arithmetic.

The receiver's ordinary prompt is never rewritten. A marker line in the
rendered prompt tells the adapter where the sender's continuous message
belongs; the worker reserves that many positions and overwrites exactly
those rows with the authorised native prefix.

Without a marker the extension places the prefix itself: immediately after
the last user-turn role token of the rendered prompt (LAYOUT_LAST_USER). The
role token is found by exact token-id match against the tokenizer's own
attested role-token id, never by a text search. When that id cannot be
attested, does not occur, or the prefix-stability proof fails, the legacy
layout (prefix prepended at absolute row 0) is used and the bounded fallback
reason is recorded. Everything here is integer arithmetic over token ids and
counts; no prompt text is retained and no token id is logged.
"""
from __future__ import annotations

from dataclasses import dataclass

from .latent_types import require

#: Ordinary ASCII marker placed by the receiver prompt. Not a model token.
PREFIX_MARKER = "<<occ3-statebegin>>"

LAYOUT_MARKER = "marker"
LAYOUT_PREPEND = "prepend"
LAYOUT_LAST_USER = "last_user"
LAYOUTS = (LAYOUT_MARKER, LAYOUT_PREPEND, LAYOUT_LAST_USER)

#: Bounded reasons a no-marker request fell back to the prepend layout.
FALLBACK_REASONS = ("no_tokenizer", "no_role_token", "role_absent", "unstable", "shape")


@dataclass(frozen=True)
class PrefixLayout:
    """Where the continuous prefix lives in the receiver sequence."""

    layout: str
    offset: int
    #: Why a no-marker request used the prepend layout; empty otherwise.
    fallback: str = ""

    def __post_init__(self):
        require(self.layout in LAYOUTS, "profile")
        require(type(self.offset) is int and 0 <= self.offset <= 2**20, "profile")
        require((self.layout != LAYOUT_PREPEND) == (self.offset > 0), "profile")
        require(self.fallback == "" or (self.layout == LAYOUT_PREPEND
                                        and self.fallback in FALLBACK_REASONS), "profile")


def _encoder(tokenizer, add_special_tokens):
    def encode(text):
        ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
        ids = list(ids) if isinstance(ids, (list, tuple)) else list(ids.tolist())
        require(all(type(value) is int for value in ids), "profile")
        return ids
    return encode


def _find_last(haystack, needle):
    """Last exact token-id occurrence, or -1. Token matching, never text.

    A single-id needle (the role token) uses one reversed C-level
    ``list.index`` scan instead of a per-position slice comparison.
    """
    if not needle or len(needle) > len(haystack):
        return -1
    if len(needle) == 1:
        try:
            return len(haystack) - 1 - list(reversed(haystack)).index(needle[0])
        except ValueError:
            return -1
    for start in range(len(haystack) - len(needle), -1, -1):
        if haystack[start:start + len(needle)] == needle:
            return start
    return -1


def resolve_prefix_layout(text, tokenizer, *, marker=PREFIX_MARKER,
                          add_special_tokens=False):
    """Resolve the insertion point of the continuous prefix in ``text``.

    The whole rendered prompt and its strict prefix must agree on the token
    ids before the marker; otherwise the tokenizer is not prefix-stable here
    and the offset cannot be attested. Refuse rather than guess.
    """
    require(type(text) is str and text, "profile")
    if marker not in text:
        return PrefixLayout(LAYOUT_PREPEND, 0)
    require(text.count(marker) == 1, "profile")
    encode = _encoder(tokenizer, add_special_tokens)
    ids, mark = encode(text), encode(marker)
    require(bool(mark), "profile")
    start = _find_last(ids, mark)
    require(start > 0, "profile")
    offset = start + len(mark)
    require(offset < len(ids), "profile")
    cut = text.index(marker) + len(marker)
    head = encode(text[:cut])
    require(len(head) == offset and ids[:offset] == head, "profile")
    return PrefixLayout(LAYOUT_MARKER, offset)


def _plain(ids):
    ids = list(ids) if isinstance(ids, (list, tuple)) else list(ids.tolist())
    require(all(type(value) is int for value in ids), "profile")
    return ids


def resolve_last_user_layout(tokenizer, role_id, *, text=None, input_ids=None,
                             add_special_tokens=False):
    """Place the prefix immediately after the last user-turn role token.

    ``role_id`` is the tokenizer's own attested user role-token id
    (span_markers.resolve_role_token_id). The position is an exact token-id
    match in the rendered prompt's ids. Returns ``(layout, pinned_ids)``:
    ``pinned_ids`` is the exact id sequence the offset was computed on (the
    caller must send these ids, not the text), or None on a prepend fallback.

    For a text prompt the same prefix-stability proof as resolve_prefix_layout
    holds: the head (the decoded ids up to and including the role token) must
    be a strict prefix of the text and must encode to exactly ids[:offset].
    A prompt that is already token ids needs no proof: the offset indexes them.
    Never raises for an ordinary prompt; every refusal is a bounded fallback.

    Limitation: the match is on the role-token id, not on the chat-template
    structure. The serving path encodes the rendered template with special
    tokens enabled, so a user message or tool result that contains the
    literal ``<|user|>`` becomes the same id, and the prefix is placed after
    that literal, inside the content, while the layout still reads
    ``last_user``. This only moves the receiver's own attached state within
    its own prompt; it never crosses requests. A structural guard would be
    just as spoofable by content, so none is attempted.
    """
    def fallback(reason):
        return PrefixLayout(LAYOUT_PREPEND, 0, fallback=reason), None
    if type(role_id) is not int or role_id < 0:
        return fallback("no_role_token")
    try:
        if text is not None:
            if type(text) is not str or not text:
                return fallback("shape")
            ids = _encoder(tokenizer, add_special_tokens)(text)
        elif input_ids is not None:
            ids = _plain(input_ids)
        else:
            return fallback("shape")
    except Exception:
        return fallback("shape")
    if not ids or len(ids) > 2**20:
        return fallback("shape")
    index = _find_last(ids, [role_id])
    if index < 0:
        return fallback("role_absent")
    offset = index + 1
    if offset >= len(ids):
        # Nothing follows the role token: no room to place the prefix before
        # the generation prompt. Not a GLM chat-template prompt.
        return fallback("shape")
    if text is not None:
        try:
            head = tokenizer.decode(ids[:offset], skip_special_tokens=False)
            stable = (type(head) is str and 0 < len(head) < len(text) and text.startswith(head)
                      and _encoder(tokenizer, add_special_tokens)(head) == ids[:offset])
        except Exception:
            stable = False
        if not stable:
            return fallback("unstable")
    return PrefixLayout(LAYOUT_LAST_USER, offset), ids


def prefix_span(tokens, offset, prefix_len, extend_len, shift=0):
    """Slice of one extend chunk covered by the continuous prefix.

    Returns ``(local_start, count, prefix_start)``: receiver rows
    ``local_start .. local_start+count-1`` of this chunk take prefix rows
    ``prefix_start .. prefix_start+count-1``. ``shift`` rotates the window by
    one row for the draft batch that replays the last accepted token.
    """
    for value in (tokens, offset, prefix_len, extend_len, shift):
        require(type(value) is int and value >= 0, "profile")
    require(tokens >= 1, "profile")
    chunk_start = prefix_len + shift
    low = max(chunk_start, offset)
    high = min(chunk_start + extend_len, offset + tokens)
    if high <= low:
        return 0, 0, 0
    return low - chunk_start, high - low, low - offset
