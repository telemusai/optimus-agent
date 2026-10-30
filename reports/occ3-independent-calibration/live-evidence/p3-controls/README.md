# P3 controls — first-live-control machinery DRAFT (OFFLINE, NOT RUN)

Owner: P3 first-live-control machinery draft. Status: **NOT RUN** — offline drafts only.
Nothing here was executed against any server; no repo file was edited. The worktree
`optimus-occ3-icalib-20261028` (branch `fix/occ3-independent-calibration-semantic-validation`)
was read only.

| file | what it is |
|---|---|
| `arms/control-T-exact.txt` | control-file bytes for arm T-exact: `texact\n` (7 bytes, sha256 `7e9d584c0ba058cdfeee7631aa61b972cf074a59379c27ff4e6526d39abf3277`) |
| `arms/control-T-norm.txt` | `tnorm\n` (6 bytes, sha256 `c19f999f06caf20191f84b61fe488bbb47c99621c266bd984b99284f6cfa3ffb`) |
| `arms/control-N-shape.txt` | `nshape\n` (7 bytes, sha256 `f21128853cc599d6da06f665a177fac177bbd3d84b41a3f7ecc60dc9fbabdc13`) |
| `arms/control-N-none.txt` | empty (0 bytes) — staged documentation that NO token is written; the control path itself must hold NO file for this arm |
| `arms/control-F-text.txt` | empty (0 bytes) — same; plus the F-text body carries no occ3 keys at all |
| `arm-schedule.json` | machine-readable schedule: 8 turns, fixed order, turn-id grammar and per-arm ranges |
| `ARM-SCHEDULE.md` | the schedule in words (which turn gets which arm, interleaved capture/attach cycles) |
| `p3_first_live_control_draft.py` | driver skeleton: submits control turns through the gateway, records arm + response metadata to `turns.jsonl`, emits the fit-exclusion list. Syntax-checked offline; never executed |
| `fit-exclusion.template.json` | `occ3.icalib.p3-fit-exclusion.v1` template (P5 exclusion) |
| `FIT-EXCLUSION-FORMAT.md` | the exclusion format spec and how P5 must apply it |

Staging note: `deploy/v3/live-instrumentation/stage_build.py` stages these five files as
`arms/control-<arm>.txt` in the attempt dir with EXACTLY these bytes (`ARM_TOKENS`), and
`stage_build.py` writes `arms/arm-order.json` documenting order, tokens, operator steps and the
P5-exclusion note. These drafts match that contract byte-for-byte.

## The exact schema the hook expects

Source: `extensions/sglang-occ3/deploy/v3/live-instrumentation/occ3_p3_arms_hook.py`.

### 1. One path, fixed at boot (never a per-turn selector)

```python
DEFAULT_CONTROL_PATH = "/tmp/occ3_p3_arm"
CONTROL_PATH_ENV = "OCC3_P3_ARM_FILE"
```

```python
def control_path():
    """The one arm control path, fixed once at boot (never a per-turn selector)."""
    raw = os.environ.get(CONTROL_PATH_ENV)
    if raw is None:
        return DEFAULT_CONTROL_PATH
    if (type(raw) is not str or not raw or raw != raw.strip() or "\x00" in raw
            or not os.path.isabs(raw) or len(raw) > 512):
        raise _Refusal("arm_path_invalid")
    return raw
```

### 2. The token parser — exactly one token, optionally one trailing newline

```python
TEXACT = "texact"
TNORM = "tnorm"
NSHAPE = "nshape"
ARMS = (TEXACT, TNORM, NSHAPE)
```

```python
def _arm_token(raw):
    """Exactly one arm token, optionally with one trailing newline, else None."""
    for token in ARMS:
        encoded = token.encode("ascii")
        if raw == encoded or raw == encoded + b"\n":
            return token
    return None
```

So the file bytes must be exactly `texact`, `tnorm` or `nshape`, or one of those plus a single
`\n`. Anything else — empty, whitespace, CRLF (`texact\r\n` parses to NO arm), a BOM, a second
newline, a list, JSON — means "no arm": the proxy returns the original aligner's result and the
shipped method runs untouched.

### 3. Size and regular-file gate

```python
MAX_CONTROL_BYTES = 64
```

```python
    if not stat.S_ISREG(info.st_mode) or size > MAX_CONTROL_BYTES:
        _CACHE.update(key=None, raw=None, digest=None, size=size, mtime_ns=mtime_ns)
        return None, None, size, mtime_ns
```

The file must be a regular file of at most 64 bytes. It is re-read on every aligner call, keyed
only on the stat tuple (size, mtime_ns, inode), so replacing it between turns always takes effect
(`read_control`, hook lines 110-141).

### 4. What each arm does (the proxy's `__call__` and `serving` property)

```python
    @property
    def serving(self):
        arm = _read_for_serving(self.__dict__["_control"])
        if arm == TNORM:
            return {"kind": "n1"}
        if arm in (TEXACT, NSHAPE):
            return None
        return getattr(self.__dict__["_original"], "serving", None)
```

```python
        if arm is None:
            return original(hidden, embeddings, target_norm)
        if getattr(embeddings, "ndim", None) != 2:
            raise RuntimeError("p3_arm_rows_invalid")
        if arm == TEXACT:
            return embeddings, None
        if arm == TNORM:
            module = importlib.import_module(CALIBRATED_MODULE)
            return module.serving_transform({"kind": "n1"}, embeddings,
                                             nu=_norm_statistic(target_norm)), None
        rows = embeddings.mean(dim=0, keepdim=True).repeat(embeddings.shape[0], 1)
        if not bool(rows.isfinite().all()):
            raise RuntimeError("p3_arm_norm_unavailable")
        return rows.contiguous(), None
```

* `texact` — the runtime-computed input-token embedding rows, unchanged.
* `tnorm` — the same rows through `serving_transform({"kind": "n1"}, ...)`; the hook converts the
  0-dim tensor `nu` to a builtin number first (`_norm_statistic`).
* `nshape` — the column mean of those rows, replicated to the same row count.
* no arm — the original aligner's own result and serving config; shipped behaviour untouched.

### 5. The proof line (arm identity from the log)

```python
def _emit(arm, digest, mtime_ns):
    LOG.warning("occ3_p3_arm=%s file_sha256=%s mtime_ns=%d",
                arm or "none", digest or "none", int(mtime_ns))
```

One sanitised line per aligner call: `occ3_p3_arm=<token|none> file_sha256=<hex|none>
mtime_ns=<int>`. No tensor, no token id, no prompt text.

### 6. Install refusals (fail closed)

The hook refuses to install (one `occ3_p3_arms install state=refused code=...` line, shipped
behaviour left in place) when: the env path is invalid (`arm_path_invalid`), the resolver is
missing/not callable/wrapped by someone else/defined elsewhere (`resolver_missing`,
`resolver_not_callable`, `resolver_already_wrapped`, `resolver_not_the_real_aligner`), the worker
rebind cannot be guaranteed (`worker_rebind_failed`, `worker_binding_not_wrapped`), or the
dispatch rebind fails (`resolver_rebind_failed`).

### 7. Mapping to the five P3 arms

| plan arm | control file at `/tmp/occ3_p3_arm` | aligner outcome | receiver body |
|---|---|---|---|
| T-exact | `texact\n` | raw input-token embedding rows | occ3 + occ3_receipt_mode |
| T-norm | `tnorm\n` | rows renormalised to `nu` (N1) | occ3 + occ3_receipt_mode |
| N-shape | `nshape\n` | column mean replicated to n rows | occ3 + occ3_receipt_mode |
| N-none | NO FILE (removed, not blanked) | not called at all — the body carries no occ3 keys, the adapter bypasses OCC-3 (chat_adapter.py:445-449) | plain |
| F-text | NO FILE | not called at all — same bypass; the private sentence rides as ordinary text between the markers | plain |

Plan 2.4 fixes the operator steps: the file is REMOVED (not blanked) before N-none and F-text, and
removed again after N-shape. The staged `control-N-none.txt` / `control-F-text.txt` (0 bytes)
exist only in the attempt dir as documentation of "no token is written"; nothing is ever copied
onto the control path for those arms.
