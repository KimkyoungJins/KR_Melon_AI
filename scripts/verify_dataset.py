"""Sanity-check Phase 1 raw labels (data/labels/) or processed output (data/processed/).

Usage:
    python scripts/verify_dataset.py --root data/labels
    python scripts/verify_dataset.py --root data/processed --processed
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLASSES, discover_from_labels  # noqa: E402


def verify_labels(root: Path) -> int:
    samples, errors = discover_from_labels(root)
    counts = Counter((s.aihub_split, s.class_name) for s in samples)

    print(f"\n[verify] labels root: {root}")
    print(f"[verify] valid={len(samples):,}  errors={len(errors)}")

    print("\n              train       val      total")
    for cls in CLASSES:
        t = counts.get(("train", cls), 0)
        v = counts.get(("val", cls), 0)
        print(f"  {cls:<10}  {t:>7,}  {v:>7,}   {t+v:>7,}")
    grand = sum(counts.values())
    print(f"  {'TOTAL':<10}  {sum(c for (s, _), c in counts.items() if s=='train'):>7,}  "
          f"{sum(c for (s, _), c in counts.items() if s=='val'):>7,}   {grand:>7,}")

    farms = sorted({s.farm_id for s in samples})
    sites = sorted({s.site_id for s in samples})
    print(f"\nfarms: {len(farms)}   sites: {len(sites)}")

    if errors:
        print(f"\nfirst 5 errors of {len(errors)}:")
        for path, reason in errors[:5]:
            print(f"  {reason}: {path}")
    return 0 if samples else 1


def verify_processed(root: Path) -> int:
    splits = ("train", "val", "test")
    print(f"\n[verify] processed root: {root}")

    issues = 0
    grand_total = 0
    for split in splits:
        d = root / split
        if not d.is_dir():
            print(f"  [fail] missing split dir: {d}")
            issues += 1
            continue
        per_cls = {}
        for c in CLASSES:
            cd = d / c
            n = len(list(cd.glob("*.jpg"))) if cd.is_dir() else -1
            per_cls[c] = n
            if n < 0:
                issues += 1
        total = sum(v for v in per_cls.values() if v >= 0)
        grand_total += total
        print(f"\n  [{split}]  total={total:,}")
        for c in CLASSES:
            n = per_cls[c]
            print(f"    {c:<10}  {n if n>=0 else 'MISSING':>7}")

    meta = root / "metadata.csv"
    groups = root / "split_groups.txt"
    plot = root / "class_distribution.png"
    print(f"\n  metadata.csv         : {'OK' if meta.exists() else 'MISSING'}")
    print(f"  split_groups.txt     : {'OK' if groups.exists() else 'MISSING'}")
    print(f"  class_distribution   : {'OK' if plot.exists() else 'MISSING'}")
    print(f"  grand total images   : {grand_total:,}")

    # Cross-check site/farm leakage in the processed metadata.
    if meta.exists():
        import csv
        site_to_splits = defaultdict(set)
        farm_to_splits = defaultdict(set)
        with meta.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                site_to_splits[row["site_id"]].add(row["split"])
                farm_to_splits[row["farm_id"]].add(row["split"])
        site_leakers = [s for s, ss in site_to_splits.items() if len(ss) > 1]
        farm_overlaps = [f for f, ss in farm_to_splits.items() if len(ss) > 1]
        # Site leakage is the hard guarantee (we used site-grouped split).
        if site_leakers:
            print(f"  [fail] site leakage in processed split: {len(site_leakers)} sites in >1 split")
            issues += 1
        else:
            print(f"  site leakage check    : OK ({len(site_to_splits)} sites, no overlap)")
        # Farm overlap is expected (documented limitation).
        print(f"  farm overlap (info)   : {len(farm_overlaps)}/{len(farm_to_splits)} farms span >1 split")

    return 0 if issues == 0 else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--processed", action="store_true")
    args = ap.parse_args()

    if not args.root.exists():
        print(f"[fail] root does not exist: {args.root}", file=sys.stderr)
        return 2
    return verify_processed(args.root) if args.processed else verify_labels(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
