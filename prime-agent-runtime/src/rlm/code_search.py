"""Present deterministic search candidates for optional host-side Jev scoring.

Retrieve with ripgrep/AST/symbol tools first. Keep the full result in a Python
variable, then call present(candidates) in a cell whose only output is this
presentation. No network calls or filesystem mutations occur here.
"""

import json
from collections.abc import Iterable, Mapping

SCHEMA = "rlm.code-search/1"
MAX_CANDIDATES = 500
MAX_BYTES = 512 * 1024
KINDS = {"file", "symbol", "grep", "reference", "test"}


def from_ripgrep(output: str) -> list[dict]:
    """Parse `rg --json` output captured through bash(); never execute a search."""
    if len(output.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Search output exceeds 512 KiB; narrow the deterministic search")
    candidates = []
    for line in output.splitlines():
        event = json.loads(line)
        if event.get("type") != "match":
            continue
        data = event["data"]
        # Binary/base64 paths and lines need explicit handling by the caller.
        path = data["path"]["text"]
        snippet = data["lines"]["text"]
        candidates.append({"kind": "grep", "path": path,
                           "line": data["line_number"], "snippet": snippet})
    return candidates


def present(candidates: Iterable[Mapping]) -> None:
    """Print a bounded, explicit candidate envelope; preserve input order/data.

    Each candidate has kind, path, optional line and snippet, and optional
    mandatory=True. Instructions and mandatory entries are pinned by the host.
    Jev can only filter when the operator enables both scoring and filtering.
    """
    items = []
    for candidate in candidates:
        if len(items) >= MAX_CANDIDATES:
            raise ValueError("At most 500 candidates; narrow the deterministic search")
        item = dict(candidate)
        if set(item) - {"kind", "path", "line", "snippet", "mandatory"}:
            raise ValueError("Unknown search candidate fields")
        if item.get("kind") not in KINDS:
            raise ValueError("Expected file, symbol, grep, reference, or test kind")
        if not isinstance(item.get("path"), str) or not item["path"] or len(item["path"]) > 1024:
            raise ValueError("Expected a bounded candidate path")
        if "line" in item and (type(item["line"]) is not int or item["line"] < 1):
            raise ValueError("Expected a positive line number")
        if not isinstance(item.get("snippet", ""), str) or len(item.get("snippet", "")) > 4096:
            raise ValueError("Expected a snippet of at most 4096 characters")
        if type(item.get("mandatory", False)) is not bool:
            raise ValueError("Expected a boolean mandatory flag")
        items.append(item)
    encoded = json.dumps({"schema": SCHEMA, "candidates": items}, ensure_ascii=False)
    if len(encoded.encode("utf-8")) > MAX_BYTES:
        raise ValueError("Candidate presentation exceeds 512 KiB")
    print(encoded)
