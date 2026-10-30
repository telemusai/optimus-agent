"""OCC-3 P3 arm instrumentation: one unattested wrap of the aligner resolver.

DECLARED INSTRUMENTATION, NOT PLUGIN SOURCE. This module is not part of the
digest-attested `occ3_sglang` package and must never be installed inside it: the
native install verifies every listed module and declines with
`source_digest_mismatch` if one is edited. It ships as two extra files in
site-packages, `occ3_p3_arms_hook.py` plus `zz_occ3_p3_arms.pth` (the .pth line
imports this module at interpreter start), and both digests are declared in the
window record.

What it does. `occ3_sglang.latent_worker` binds `aligner_for` at import
(`from .alignment_dispatch import aligner_for, ...`) and calls that module-level
name once per capture turn, so patching `alignment_dispatch.aligner_for` alone
would not select an arm. The hook wraps the resolver in both places: it sets
`occ3_sglang.alignment_dispatch.aligner_for` and rebinds
`occ3_sglang.latent_worker.aligner_for` when that module is already imported.
The wrapped resolver returns a proxy aligner over the real one that chooses its
arm per call from a control file read fresh on every call.

Arms (live-first-control-plan.md section 2.4; only the resolver's own arguments
differ between them):

* `texact`  - return the runtime-computed input-token embedding rows unchanged.
* `tnorm`   - return the same rows renormalised to the vocabulary norm statistic
              nu (`statebridge_calibrated.serving_transform({"kind": "n1"}, ...)`).
              The worker measures nu as a 0-dim tensor and that function takes a
              builtin number (it refuses a tensor with `profile`), so the arm
              converts it once with `_norm_statistic`.
* `nshape`  - labelled null/mean control: the column mean of those rows,
              replicated to the same row count, dtype and positions.

No arm file, an empty/oversized/unreadable file, or any other content means "no
arm": the proxy returns the original aligner's own result and exposes the
original aligner's own `serving` config, so the shipped method runs untouched.

Fail closed. If the hook cannot wrap the real resolver (missing, not callable,
already wrapped by someone else, defined elsewhere), or cannot guarantee the
`latent_worker` rebinding, or the control-file path declared at boot is invalid,
it refuses to install: it logs one `occ3_p3_arms install state=refused code=...`
line, records the refusal in `STATE`, restores the original attribute and leaves
the shipped behaviour in place. It never proceeds silently.

Arm identity is provable from the log: one sanitised line per aligner call,
`occ3_p3_arm=<token|none> file_sha256=<hex|none> mtime_ns=<int>`. No tensor, no
token id and no prompt text is ever logged.
"""
from __future__ import annotations

import functools
import hashlib
import importlib
import importlib.abc
import importlib.util
import logging
import math
import os
import stat
import sys

TEXACT = "texact"
TNORM = "tnorm"
NSHAPE = "nshape"
ARMS = (TEXACT, TNORM, NSHAPE)
DEFAULT_CONTROL_PATH = "/tmp/occ3_p3_arm"
CONTROL_PATH_ENV = "OCC3_P3_ARM_FILE"
MAX_CONTROL_BYTES = 64
DISPATCH_MODULE = "occ3_sglang.alignment_dispatch"
WORKER_MODULE = "occ3_sglang.latent_worker"
CALIBRATED_MODULE = "occ3_sglang.statebridge_calibrated"
RESOLVER_NAME = "aligner_for"

LOG = logging.getLogger("occ3_p3_arms")

STATE = {"state": "pending", "code": None, "detail": None, "control_path": None,
         "armed": False, "patched": []}

_CACHE = {"key": None, "raw": None, "digest": None, "size": 0, "mtime_ns": 0}


class _Refusal(Exception):
    """Internal control flow: one fixed refusal code, never partial install."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


# -- arm control file ---------------------------------------------------------

def control_path():
    """The one arm control path, fixed once at boot (never a per-turn selector)."""
    raw = os.environ.get(CONTROL_PATH_ENV)
    if raw is None:
        return DEFAULT_CONTROL_PATH
    if (type(raw) is not str or not raw or raw != raw.strip() or "\x00" in raw
            or not os.path.isabs(raw) or len(raw) > 512):
        raise _Refusal("arm_path_invalid")
    return raw


def _arm_token(raw):
    """Exactly one arm token, optionally with one trailing newline, else None."""
    for token in ARMS:
        encoded = token.encode("ascii")
        if raw == encoded or raw == encoded + b"\n":
            return token
    return None


def read_control(path=None):
    """(token|None, sha256|None, size, mtime_ns) of the arm file, re-read per call.

    The stat tuple (size, mtime_ns, inode) is the only cache key, so a file
    replaced with different content is always re-read; content is reused only
    while that tuple is unchanged.
    """
    path = control_path() if path is None else path
    try:
        info = os.stat(path)
    except OSError:
        _CACHE.update(key=None, raw=None, digest=None, size=0, mtime_ns=0)
        return None, None, 0, 0
    mtime_ns, size = int(info.st_mtime_ns), int(info.st_size)
    if not stat.S_ISREG(info.st_mode) or size > MAX_CONTROL_BYTES:
        _CACHE.update(key=None, raw=None, digest=None, size=size, mtime_ns=mtime_ns)
        return None, None, size, mtime_ns
    key = (size, mtime_ns, int(getattr(info, "st_ino", 0)))
    if key == _CACHE["key"] and _CACHE["raw"] is not None:
        return _arm_token(_CACHE["raw"]), _CACHE["digest"], size, mtime_ns
    try:
        with open(path, "rb") as handle:
            raw = handle.read(MAX_CONTROL_BYTES + 1)
    except OSError:
        _CACHE.update(key=None, raw=None, digest=None, size=size, mtime_ns=mtime_ns)
        return None, None, size, mtime_ns
    if len(raw) > MAX_CONTROL_BYTES:
        _CACHE.update(key=None, raw=None, digest=None, size=len(raw), mtime_ns=mtime_ns)
        return None, None, len(raw), mtime_ns
    digest = hashlib.sha256(raw).hexdigest()
    _CACHE.update(key=key, raw=raw, digest=digest, size=len(raw), mtime_ns=mtime_ns)
    return _arm_token(raw), digest, len(raw), mtime_ns


def _norm_statistic(target_norm):
    """The vocabulary norm statistic as the builtin number serving_transform takes.

    The worker measures nu with `mean_embedding_norm`, which returns a 0-dim
    tensor, and `serving_transform` requires a builtin int/float (it refuses a
    tensor with `profile`). The conversion happens here, once, and a value that
    cannot be a positive finite number declines the turn instead of silently
    serving the wrong scale.
    """
    if target_norm is None or isinstance(target_norm, (bool, str, bytes)):
        raise RuntimeError("p3_arm_norm_unavailable")
    item = getattr(target_norm, "item", None)
    try:
        value = float(item() if item is not None else target_norm)
    except Exception:
        raise RuntimeError("p3_arm_norm_unavailable") from None
    if not (math.isfinite(value) and value > 0.0):
        raise RuntimeError("p3_arm_norm_unavailable")
    return value


def _read_for_serving(path):
    """The `serving` property must never raise inside the worker's renorm check."""
    try:
        return read_control(path)[0]
    except Exception:
        return None


def _emit(arm, digest, mtime_ns):
    LOG.warning("occ3_p3_arm=%s file_sha256=%s mtime_ns=%d",
                arm or "none", digest or "none", int(mtime_ns))


# -- the arm proxy ------------------------------------------------------------

class _ArmAligner:
    """The real aligner behind a per-call arm selector."""

    __occ3_p3_wrapped__ = True

    def __init__(self, original, control, method=""):
        self.__dict__["_original"] = original
        self.__dict__["_control"] = control
        self.__dict__["_method"] = method

    def __getattr__(self, name):
        return getattr(self.__dict__["_original"], name)

    def __repr__(self):
        return "<occ3_p3_arm aligner method=%r original=%r>" % (
            self.__dict__["_method"], self.__dict__["_original"])

    @property
    def serving(self):
        """Per-call serving config: the arm decides it, the worker reads it first.

        The worker measures the vocabulary norm statistic only when this is not
        None (`_serving_renorm`), so a stored value would spend or skip that
        pass for the wrong arm.
        """
        arm = _read_for_serving(self.__dict__["_control"])
        if arm == TNORM:
            return {"kind": "n1"}
        if arm in (TEXACT, NSHAPE):
            return None
        return getattr(self.__dict__["_original"], "serving", None)

    @property
    def artifact_digest(self):
        return getattr(self.__dict__["_original"], "artifact_digest", "")

    @property
    def rows(self):
        return getattr(self.__dict__["_original"], "rows", None)

    def __call__(self, hidden, embeddings, target_norm):
        original, control = self.__dict__["_original"], self.__dict__["_control"]
        arm, digest, _size, mtime_ns = read_control(control)
        _emit(arm, digest, mtime_ns)
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


# -- resolver wrap ------------------------------------------------------------

def _make_resolver(original, control):
    """The resolver that replaces aligner_for: same signature, wrapped result."""

    @functools.wraps(original)
    def aligner_for(profile, artifact=None):
        aligner = original(profile, artifact)
        return _ArmAligner(aligner, control, getattr(profile, "method", ""))

    aligner_for.__occ3_p3_wrapped__ = True
    aligner_for.__occ3_p3_original__ = original
    aligner_for.__occ3_p3_control__ = control
    return aligner_for


def _mark_patched(module_name):
    if module_name not in STATE["patched"]:
        STATE["patched"].append(module_name)
        LOG.warning("occ3_p3_arms install state=wrapped target=%s", module_name)


def _rebind_worker():
    """Rebind the module-level name latent_worker bound at import, or refuse."""
    worker = sys.modules.get(WORKER_MODULE)
    if worker is None:
        # Not imported yet: it binds from alignment_dispatch after our patch,
        # which is the guaranteed order.
        return
    dispatch = sys.modules.get(DISPATCH_MODULE)
    wrapper = getattr(dispatch, RESOLVER_NAME, None)
    if getattr(wrapper, "__occ3_p3_wrapped__", False) is not True:
        raise _Refusal("resolver_not_wrapped")
    if getattr(worker, RESOLVER_NAME, None) is wrapper:
        _mark_patched(WORKER_MODULE)
        return
    try:
        worker.__dict__[RESOLVER_NAME] = wrapper
    except Exception:
        raise _Refusal("worker_rebind_failed") from None
    if getattr(worker, RESOLVER_NAME, None) is not wrapper:
        raise _Refusal("worker_rebind_failed")
    _mark_patched(WORKER_MODULE)


def _patch_dispatch(module):
    """Wrap the real resolver in alignment_dispatch, transactionally."""
    original = getattr(module, RESOLVER_NAME, None)
    if original is None:
        raise _Refusal("resolver_missing")
    if not callable(original):
        raise _Refusal("resolver_not_callable")
    if getattr(original, "__occ3_p3_wrapped__", False) is True:
        # Our own earlier wrap, or a foreign wrap this hook cannot vouch for.
        if getattr(original, "__occ3_p3_original__", None) is not None:
            _mark_patched(DISPATCH_MODULE)
            _rebind_worker()
            return
        raise _Refusal("resolver_already_wrapped")
    if (getattr(original, "__module__", "") != DISPATCH_MODULE
            or getattr(original, "__qualname__", "") != RESOLVER_NAME):
        raise _Refusal("resolver_not_the_real_aligner")
    wrapper = _make_resolver(original, STATE["control_path"])
    module.__dict__[RESOLVER_NAME] = wrapper
    try:
        if getattr(module, RESOLVER_NAME) is not wrapper:
            raise _Refusal("resolver_rebind_failed")
        _rebind_worker()
    except BaseException:
        module.__dict__[RESOLVER_NAME] = original
        raise
    _mark_patched(DISPATCH_MODULE)


def _verify_worker(module):
    """Prove the name latent_worker will call is the wrapped one."""
    dispatch = sys.modules.get(DISPATCH_MODULE)
    wrapper = getattr(dispatch, RESOLVER_NAME, None) if dispatch is not None else None
    # latent_worker imports alignment_dispatch before it binds the name, so an
    # unpatched dispatch here means the hook cannot prove the call site. That
    # is a refusal, never a silent pass.
    if getattr(wrapper, "__occ3_p3_wrapped__", False) is not True:
        raise _Refusal("worker_binding_not_wrapped")
    if getattr(module, RESOLVER_NAME, None) is not wrapper:
        raise _Refusal("worker_binding_not_wrapped")
    _mark_patched(WORKER_MODULE)


def _patch_loaded():
    dispatch = sys.modules.get(DISPATCH_MODULE)
    if dispatch is not None:
        _patch_dispatch(dispatch)
        return True
    return False


# -- import-order hooks -------------------------------------------------------

class _WrappedLoader(importlib.abc.Loader):
    """Delegate to the real loader, then run one callback after module exec."""

    def __init__(self, loader, callback):
        self._loader = loader
        self._callback = callback

    def create_module(self, spec):
        return self._loader.create_module(spec)

    def exec_module(self, module):
        self._loader.exec_module(module)
        try:
            self._callback(module)
        except _Refusal as error:
            _refuse(error.code)
        except Exception as error:
            _refuse("hook_" + type(error).__name__)

    def __getattr__(self, name):
        return getattr(self.__dict__["_loader"], name)


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Wrap the loaders of the two modules the arm has to reach."""

    def __init__(self, callbacks):
        self._callbacks = dict(callbacks)

    def find_spec(self, fullname, path=None, target=None):
        callback = self._callbacks.get(fullname)
        if callback is None:
            return None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            find_spec = getattr(finder, "find_spec", None)
            if find_spec is None:
                continue
            try:
                spec = find_spec(fullname, path, target)
            except Exception:
                continue
            if spec is None:
                continue
            if spec.loader is not None and not isinstance(spec.loader, _WrappedLoader):
                spec.loader = _WrappedLoader(spec.loader, callback)
            return spec
        return None


def _install_finder():
    for finder in list(sys.meta_path):
        if isinstance(finder, _PostImportFinder):
            return
    sys.meta_path.insert(0, _PostImportFinder({DISPATCH_MODULE: _patch_dispatch,
                                               WORKER_MODULE: _verify_worker}))
    STATE["armed"] = True


def _remove_finders():
    for finder in list(sys.meta_path):
        if isinstance(finder, _PostImportFinder):
            sys.meta_path.remove(finder)
    STATE["armed"] = False


def _refuse(code, detail=None):
    """Record one refusal, disarm, and leave the shipped behaviour untouched."""
    STATE.update(state="refused", code=code, detail=detail)
    _remove_finders()
    LOG.error("occ3_p3_arms install state=refused code=%s control=%s",
              code, STATE["control_path"])
    return dict(STATE)


def install(control=None):
    """Arm the hook once. Idempotent; a refusal is sticky and loud."""
    if STATE["state"] in ("installed", "refused"):
        return dict(STATE)
    try:
        STATE["control_path"] = control if control is not None else control_path()
    except _Refusal as error:
        return _refuse(error.code)
    try:
        _install_finder()
        _patch_loaded()
    except _Refusal as error:
        return _refuse(error.code)
    except Exception as error:
        return _refuse("install_" + type(error).__name__)
    STATE.update(state="installed")
    LOG.warning("occ3_p3_arms install state=installed control=%s armed=%s patched=%s",
                STATE["control_path"], STATE["armed"], ",".join(STATE["patched"]) or "deferred")
    return dict(STATE)


def status():
    """Scalar-only install state for the window record."""
    return {"state": STATE["state"], "code": STATE["code"], "armed": STATE["armed"],
            "control_path": STATE["control_path"], "patched": list(STATE["patched"]),
            "arms": list(ARMS)}


def _autostart():
    try:
        install()
    except Exception:
        # A hook must never break interpreter start; the refusal is already
        # recorded and logged by install().
        pass


if __name__ == "occ3_p3_arms_hook":
    _autostart()
