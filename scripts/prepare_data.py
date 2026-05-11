"""Phase 1: build a balanced 5-class subset from AIHub zips, with farm-grouped split.

Inputs (no extraction needed):
  - data/labels/{train,val}/<class>/*.json   (labels — already extracted)
  - data/247.지능형 스마트팜(참외) 데이터/.../{TS,VS}_<class>.zip  (images — read directly)

Outputs:
  - data/processed/{train,val,test}/<class>/<stem>.jpg
  - data/processed/metadata.csv
  - data/processed/class_distribution.png
  - data/processed/split_groups.txt    (which farms went into which split)

Why site-grouped (not farm-grouped): AIHub's split is random-by-image so all 12
farms appear in both splits — that inflates accuracy. A pure farm-grouped split
would be cleaner, but 흰가루병 has 92% of its images in a single farm (FM05),
so farm-grouping destroys class coverage. We compromise: split by `indvd_code`
(250 sites, much finer than 12 farms) using sklearn `StratifiedGroupKFold` —
class-stratified, site-grouped. Farm-level leakage is partially reduced and we
report exactly which farms span splits, so the limitation is visible.

Usage:
  python scripts/prepare_data.py --labels-root data/labels \\
      --aihub-root "data/247.지능형 스마트팜(참외) 데이터/01-1.정식개방데이터" \\
      --out-root data/processed --target-size 160 --per-class 2500 --seed 42
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    CLASSES, DEFAULT_AIHUB_ROOT, Sample,
    discover_from_labels, get_image_zip, letterbox, open_image_from_bytes, read_image_bytes,
)

SPLIT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def site_grouped_split(
    samples: list[Sample],
    seed: int,
) -> dict[str, list[Sample]]:
    """Split by `indvd_code` (site) with class stratification.

    StratifiedGroupKFold(n_splits=10) gives ~10% folds. Take fold 0 → test,
    fold 1 → val, the rest → train (=80/10/10).
    """
    from sklearn.model_selection import StratifiedGroupKFold

    classes = [s.class_name for s in samples]
    sites = [s.site_id for s in samples]

    skgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=seed)
    folds = list(skgkf.split(samples, classes, sites))
    test_idx = set(folds[0][1])
    val_idx = set(folds[1][1])

    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for i, s in enumerate(samples):
        if i in test_idx:
            out["test"].append(s)
        elif i in val_idx:
            out["val"].append(s)
        else:
            out["train"].append(s)

    # Diagnostics: site count, farm count, class distribution, farm overlap.
    print("\n[split] site-grouped split (StratifiedGroupKFold n=10, fold0=test, fold1=val):")
    farms_in_split: dict[str, set[str]] = {}
    for sp in ("train", "val", "test"):
        items = out[sp]
        sites_here = {s.site_id for s in items}
        farms_here = {s.farm_id for s in items}
        farms_in_split[sp] = farms_here
        cls_count = Counter(s.class_name for s in items)
        cls_str = "  ".join(f"{c}={cls_count.get(c, 0)}" for c in CLASSES)
        print(f"  {sp:<5} {len(items):>7,} imgs  sites={len(sites_here):>3}  farms={len(farms_here):>2}")
        print(f"        {cls_str}")

    farm_overlap = (farms_in_split["train"] & farms_in_split["val"] |
                    farms_in_split["train"] & farms_in_split["test"] |
                    farms_in_split["val"] & farms_in_split["test"])
    if farm_overlap:
        print(f"  [note] {len(farm_overlap)} farm(s) span multiple splits "
              f"(expected with site-grouping): {sorted(farm_overlap)}")
    return out


def cap_per_class(
    split_samples: dict[str, list[Sample]],
    per_class_cap: int,
    seed: int,
) -> dict[str, list[Sample]]:
    """Within each split, cap each class to per_class_cap × split_ratio.

    Smaller splits get fewer images per class so train/val/test stay roughly
    proportional in size.
    """
    rng = random.Random(seed + 1)

    out: dict[str, list[Sample]] = {sp: [] for sp in split_samples}
    for sp, items in split_samples.items():
        cap = max(1, int(round(per_class_cap * SPLIT_RATIOS[sp])))
        per_class = defaultdict(list)
        for s in items:
            per_class[s.class_name].append(s)

        for cls in CLASSES:
            pool = per_class.get(cls, [])
            rng.shuffle(pool)
            taken = pool[:cap]
            out[sp].extend(taken)
            if len(taken) < cap:
                print(f"  [warn] {sp}/{cls}: only {len(taken)} available (wanted {cap})")
    return out


def process_split(
    samples: list[Sample],
    aihub_root: Path,
    out_root: Path,
    split: str,
    target_size: int,
) -> list[dict[str, str]]:
    """Open each (aihub_split, class) zip once, extract+letterbox+save its assigned images."""
    from PIL import UnidentifiedImageError
    from tqdm import tqdm
    import zipfile

    # Group by (aihub_split, class) so each zip is opened exactly once.
    by_zip: dict[tuple[str, str], list[Sample]] = defaultdict(list)
    for s in samples:
        by_zip[(s.aihub_split, s.class_name)].append(s)

    rows: list[dict[str, str]] = []
    failed = 0

    for (aihub_split, cls), batch in by_zip.items():
        zip_path = get_image_zip(aihub_root, aihub_split, cls)
        out_dir = out_root / split / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path) as zf:
            for sample in tqdm(batch, desc=f"{split}/{cls}/{aihub_split}", unit="img"):
                stem = Path(sample.image_name).stem
                dst = out_dir / f"{stem}.jpg"
                try:
                    raw = read_image_bytes(zf, sample.image_name)
                    img = open_image_from_bytes(raw).convert("RGB")
                    img = letterbox(img, target_size)
                    img.save(dst, format="JPEG", quality=92, optimize=True)
                except (KeyError, OSError, UnidentifiedImageError) as e:
                    failed += 1
                    if failed <= 5:
                        print(f"  [skip] {sample.image_name}: {e}", file=sys.stderr)
                    continue

                rows.append({
                    "image_path": str(dst.relative_to(out_root)),
                    "class_name": cls,
                    "split": split,
                    "farm_id": sample.farm_id,
                    "site_id": sample.site_id,
                    "day_section": sample.day_section,
                    "aihub_split": aihub_split,
                })

    if failed:
        print(f"  [warn] {split}: skipped {failed} image(s)")
    return rows


def write_metadata(rows: list[dict[str, str]], out_root: Path) -> None:
    fields = ["image_path", "class_name", "split", "farm_id", "site_id", "day_section", "aihub_split"]
    csv_path = out_root / "metadata.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[meta] {csv_path}: {len(rows)} rows")


def write_split_groups(split_assignment: dict[str, list[Sample]], out_root: Path) -> None:
    """Record split-by-site / split-by-farm composition for forensic reporting."""
    txt = out_root / "split_groups.txt"
    with txt.open("w", encoding="utf-8") as f:
        for sp in ("train", "val", "test"):
            items = split_assignment.get(sp, [])
            sites = sorted({s.site_id for s in items})
            farms = sorted({s.farm_id for s in items})
            f.write(f"{sp}: {len(items)} imgs, {len(sites)} sites, {len(farms)} farms\n")
            f.write(f"  farms: {', '.join(farms)}\n")
        # Cross-overlap report
        sites_per_split = {sp: {s.site_id for s in split_assignment[sp]} for sp in split_assignment}
        farms_per_split = {sp: {s.farm_id for s in split_assignment[sp]} for sp in split_assignment}
        f.write("\n# leakage checks\n")
        f.write(f"site overlap train∩val:  {len(sites_per_split['train'] & sites_per_split['val'])}\n")
        f.write(f"site overlap train∩test: {len(sites_per_split['train'] & sites_per_split['test'])}\n")
        f.write(f"site overlap val∩test:   {len(sites_per_split['val'] & sites_per_split['test'])}\n")
        f.write(f"farm overlap (informational): "
                f"train∩val={len(farms_per_split['train'] & farms_per_split['val'])}, "
                f"train∩test={len(farms_per_split['train'] & farms_per_split['test'])}, "
                f"val∩test={len(farms_per_split['val'] & farms_per_split['test'])}\n")
    print(f"[meta] {txt}")


def plot_distribution(rows: list[dict[str, str]], out_root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Try to find a CJK font on the system; fall back silently.
    for fp in (
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break

    counts = {sp: Counter() for sp in ("train", "val", "test")}
    for r in rows:
        counts[r["split"]][r["class_name"]] += 1

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = list(range(len(CLASSES)))
    width = 0.27
    for i, sp in enumerate(("train", "val", "test")):
        ax.bar(
            [xi + (i - 1) * width for xi in x],
            [counts[sp].get(c, 0) for c in CLASSES],
            width=width, label=sp,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, rotation=15)
    ax.set_ylabel("# images")
    ax.set_title("Class distribution per split (site-grouped)")
    ax.legend()
    fig.tight_layout()
    out = out_root / "class_distribution.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"[plot] {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", type=Path, default=Path("data/labels"))
    ap.add_argument("--aihub-root", type=Path, default=DEFAULT_AIHUB_ROOT)
    ap.add_argument("--out-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--target-size", type=int, default=160)
    ap.add_argument("--per-class", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not args.labels_root.exists():
        print(f"[fail] labels root not found: {args.labels_root}", file=sys.stderr)
        return 2
    if not args.aihub_root.exists():
        print(f"[fail] aihub root not found: {args.aihub_root}", file=sys.stderr)
        return 2
    args.out_root.mkdir(parents=True, exist_ok=True)

    print(f"[scan] labels: {args.labels_root}")
    samples, errors = discover_from_labels(args.labels_root)
    print(f"[scan] valid={len(samples):,}  errors={len(errors)}")
    if errors[:3]:
        for p, msg in errors[:3]:
            print(f"  [err] {msg}: {p}")
    if not samples:
        return 1

    splits = site_grouped_split(samples, seed=args.seed)
    splits = cap_per_class(splits, args.per_class, seed=args.seed)

    print()
    for sp in ("train", "val", "test"):
        cls_count = Counter(s.class_name for s in splits[sp])
        print(f"[split] {sp:<5} ({len(splits[sp]):>5,}) — {dict(cls_count)}")

    print(f"\n[process] reading images directly from zips under {args.aihub_root}")
    rows: list[dict[str, str]] = []
    for sp in ("train", "val", "test"):
        rows.extend(process_split(splits[sp], args.aihub_root, args.out_root, sp, args.target_size))

    write_metadata(rows, args.out_root)
    write_split_groups(splits, args.out_root)
    plot_distribution(rows, args.out_root)

    print("\n[done] processed dataset ready.")
    print(f"  next: python scripts/verify_dataset.py --root {args.out_root} --processed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
