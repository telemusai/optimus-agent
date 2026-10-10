#!/usr/bin/env python
"""Entry point: regenerate the common-format benchmark files.

Default (no flags): dev subsets + dev/full manifests for all four families.
  longmemeval   100 envs/questions      (data/common/longmemeval/)
  locomo        10 envs, 300 questions  (data/common/locomo/)
  lme_v2        2 envs, 120 questions  (data/common/lme_v2/)
  locomo_plus   120 envs/questions      (data/common/locomo_plus/)

--full ADDITIONALLY emits the complete env/qs sets under <family>/full/
for the supplied inputs. --full first regenerates dev outputs in the family
roots, then writes the full sets. Use a fresh dedicated output root.

Deterministic: seed 20261007 (bench.converters.common.SEED); sampling uses
md5(seed:key) ordering -- see common.py.

Usage:
  python -m bench.converters.run_converters                     # dev subsets
  python -m bench.converters.run_converters --family locomo     # one family
  python -m bench.converters.run_converters --full              # dev + full sets
"""

from __future__ import annotations

import argparse
import time

from . import longmemeval, locomo, lme_v2, locomo_plus
from .common import configure_paths

FAMILIES = {
    "longmemeval": longmemeval,
    "locomo": locomo,
    "lme_v2": lme_v2,
    "locomo_plus": locomo_plus,
}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=sorted(FAMILIES), default=None,
                    help="convert only this family (default: all)")
    ap.add_argument("--full", action="store_true",
                    help="also emit the complete env/qs sets under <family>/full/")
    ap.add_argument("--data-root", help="raw input directory (or MEMORY_BENCH_DATA_ROOT)")
    ap.add_argument("--out-dir", help="common output root (or MEMORY_BENCH_COMMON_ROOT)")
    args = ap.parse_args(argv)
    configure_paths(args.data_root, args.out_dir)

    fams = [args.family] if args.family else list(FAMILIES)
    for name in fams:
        mod = FAMILIES[name]
        t0 = time.time()
        manifest = mod.build(dev_only=not args.full)
        dt = time.time() - t0
        print(f"[{name}] envs={manifest['n_envs']} questions={manifest['n_questions']} "
              f"chars={manifest['total_env_chars']:,} "
              f"~tokens={manifest['estimated_ingest_tokens']:,} ({dt:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
