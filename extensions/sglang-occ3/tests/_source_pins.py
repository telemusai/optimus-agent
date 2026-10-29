"""Shared source-root pinning for the offline OCC-3 suites, plus the runbook.

RUNBOOK - source-attested suites, pinned roots and verification commands
=========================================================================

Two different honesty claims need two different source roots, so two
environment variables name them. One variable cannot honestly serve both:

* ``OCC3_SGLANG_SOURCE_BASE`` - the PRISTINE, unpatched SGLang checkout the
  capture patch was computed against. Pinned tree: SGLang v0.5.21, commit
  ``e00930c5489053f26d86b179cee0d087f846acbb``. Default root:
  ``<worktree-sibling>/occ3-latent-validation/references/sglang``.
  Used by the suites that attest upstream arithmetic and patch fit:
  ``tests/test_predictor_alignment.py`` (via this module).
  The capture-manifest ``before_sha256`` values must match this tree.

* ``OCC3_SGLANG_SOURCE`` - the LIVE patched source of the deployed image.
  Used by the suites that attest what actually serves:
  ``tests/test_chat_attachment.py`` (and ``tests/test_chat_source_capture.py``
  through it) plus ``tests/test_native_install.py``. Their pinned hashes match
  ``C:/Users/openclawuser/occ3-live-extract/sglang-src`` and
  ``.../occ3-campaign/deps/sglang-v0.5.21``. This module NEVER reads that
  variable: the live tree is patched and partial, and attesting it as the
  pristine upstream checkout would be false.

Verification commands (venv python, from ``extensions/sglang-occ3``):

    OCC3_SGLANG_SOURCE_BASE=C:/Users/openclawuser/occ3-latent-validation/references/sglang \
        PYTHONPATH=. <python> -B -m pytest -p no:cacheprovider -q tests/test_predictor_alignment.py

    OCC3_SGLANG_SOURCE=C:/Users/openclawuser/occ3-live-extract/sglang-src \
        PYTHONPATH=. <python> -B -m pytest -p no:cacheprovider -q \
        tests/test_chat_attachment.py tests/test_chat_source_capture.py

Both may be set in one environment: the predictor suite reads only
``OCC3_SGLANG_SOURCE_BASE`` and the chat suites read only
``OCC3_SGLANG_SOURCE``. When the pristine root is absent, every
source-attested predictor test SKIPS with a reason naming
``OCC3_SGLANG_SOURCE_BASE``; no pinned hash or assertion is weakened.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

PINNED_SGLANG_COMMIT = "e00930c5489053f26d86b179cee0d087f846acbb"
PINNED_SGLANG_VERSION = "0.5.21"

_HERE = Path(__file__).resolve()
EXTENSION = _HERE.parents[1]                    # extensions/sglang-occ3
DEFAULT_PRISTINE_ROOT = _HERE.parents[3].parent / "occ3-latent-validation" / "references" / "sglang"


def pristine_root(env=None) -> Path:
    """Pristine pinned checkout root; never the live source root.

    Resolution order: ``OCC3_SGLANG_SOURCE_BASE``, then the sibling reference
    checkout. ``OCC3_SGLANG_SOURCE`` is deliberately not consulted: that
    variable names the live patched source attested by the chat suites.
    """
    env = os.environ if env is None else env
    return Path(env.get("OCC3_SGLANG_SOURCE_BASE") or DEFAULT_PRISTINE_ROOT)


def srt_dir(root) -> Path:
    return Path(root) / "python/sglang/srt"


def absent_reason(root: Path) -> str:
    return (
        f"pinned SGLang v{PINNED_SGLANG_VERSION} (commit {PINNED_SGLANG_COMMIT}) "
        f"checkout absent at {root}; set OCC3_SGLANG_SOURCE_BASE"
    )


def needs_pristine_source(env=None):
    root = pristine_root(env)
    return pytest.mark.skipif(not srt_dir(root).is_dir(), reason=absent_reason(root))


def pinned(path, env=None) -> str:
    """Text of a pinned pristine-source file, or an explicit skip."""
    path = Path(path)
    if not path.is_file():
        pytest.skip(
            f"pinned SGLang source absent: {path} ({absent_reason(pristine_root(env))})"
        )
    return path.read_text(encoding="utf-8")
