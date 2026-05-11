"""Aggregate stats over all extracted JSON labels.

Reports:
- per-class counts (train/val)
- image dimension distribution
- day vs night
- per-farm and per-site distribution (for split-leakage check)
- environmental sensor ranges (basic sanity)

Usage:
    python scripts/analyze_dataset.py --labels-root data/labels
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, pstdev


def iter_jsons(labels_root: Path):
    for split_dir in sorted(labels_root.iterdir()):
        if not split_dir.is_dir():
            continue
        for cls_dir in sorted(split_dir.iterdir()):
            if not cls_dir.is_dir():
                continue
            for jp in cls_dir.rglob("*.json"):
                yield split_dir.name, cls_dir.name, jp


def fmt_pct(n: int, total: int) -> str:
    return f"{n:>7,}  ({n/total*100:5.1f}%)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-root", type=Path, default=Path("data/labels"))
    ap.add_argument("--out", type=Path, default=Path("docs/dataset_analysis.md"))
    args = ap.parse_args()

    counts: dict[tuple[str, str], int] = Counter()
    dims: list[tuple[int, int]] = []
    day_section_by_class: dict[str, Counter] = defaultdict(Counter)
    farm_by_class: dict[str, Counter] = defaultdict(Counter)
    site_by_class: dict[str, Counter] = defaultdict(Counter)
    farm_split: dict[str, set] = defaultdict(set)  # farm → {splits it appears in}
    env_inr_tp: list[float] = []
    env_inr_hd: list[float] = []
    bad: list[tuple[Path, str]] = []

    for split, cls, jp in iter_jsons(args.labels_root):
        try:
            with jp.open(encoding="utf-8") as f:
                d = json.load(f)
            img = d["image"]
            corps = d.get("corps_info", {})
            envs = d.get("environments", []) or []
        except Exception as e:
            bad.append((jp, str(e)))
            continue

        counts[(split, cls)] += 1
        dims.append((int(img.get("image_wdth", 0)), int(img.get("image_hght", 0))))
        day_section_by_class[cls][img.get("day_section", "?")] += 1
        farm = corps.get("frm_id", "?")
        site = corps.get("indvd_code", "?")
        farm_by_class[cls][farm] += 1
        site_by_class[cls][site] += 1
        farm_split[farm].add(split)
        for e in envs[:1]:  # one record per JSON is enough for ranges
            try:
                env_inr_tp.append(float(e.get("inr_tp", "nan")))
                env_inr_hd.append(float(e.get("inr_hd", "nan")))
            except (TypeError, ValueError):
                pass

    # --- aggregate ---
    classes = sorted({c for _, c in counts})
    splits = sorted({s for s, _ in counts})

    lines: list[str] = []
    p = lines.append

    p("# Dataset analysis — AIHub 지능형 스마트팜(참외)\n")
    p(f"Source: `{args.labels_root}`\n")
    total = sum(counts.values())
    p(f"**Total labeled images: {total:,}**\n")

    # 1. class counts
    p("## 1. Class distribution\n")
    p("| class | train | val | total |")
    p("|---|---:|---:|---:|")
    cls_total = Counter()
    for cls in classes:
        row = [cls]
        sub = 0
        for sp in splits:
            v = counts[(sp, cls)]
            row.append(f"{v:,}")
            sub += v
        cls_total[cls] = sub
        row.append(f"**{sub:,}**")
        p("| " + " | ".join(row) + " |")
    p(f"| **total** | {sum(counts[(s, c)] for s in splits for c in classes if s == 'train'):,} | "
      f"{sum(counts[(s, c)] for s in splits for c in classes if s == 'val'):,} | "
      f"**{total:,}** |\n")

    smallest_total = min(cls_total.values())
    largest_total = max(cls_total.values())
    p(f"Imbalance ratio (max/min) = {largest_total/smallest_total:.2f}× "
      f"— smallest class: {min(cls_total, key=cls_total.get)} ({smallest_total:,}).\n")

    # 2. image dimensions
    p("## 2. Image dimensions\n")
    if dims:
        ws = [w for w, _ in dims if w > 0]
        hs = [h for _, h in dims if h > 0]
        wh_pairs = Counter(dims)
        p(f"- width  : min={min(ws)}, median={int(median(ws))}, mean={mean(ws):.0f}, max={max(ws)}")
        p(f"- height : min={min(hs)}, median={int(median(hs))}, mean={mean(hs):.0f}, max={max(hs)}")
        p("\nTop unique (W×H) combinations:\n")
        for (w, h), n in wh_pairs.most_common(8):
            p(f"  - {w}×{h}  →  {n:,}  ({n/total*100:.1f}%)")
        p("")

    # 3. day vs night
    p("## 3. Day vs night (image.day_section)\n")
    p("| class | 주간 | 야간 | 기타 |")
    p("|---|---:|---:|---:|")
    for cls in classes:
        c = day_section_by_class[cls]
        d = c.get("주간", 0)
        n = c.get("야간", 0)
        other = sum(v for k, v in c.items() if k not in ("주간", "야간"))
        p(f"| {cls} | {d:,} | {n:,} | {other:,} |")
    p("")

    # 4. farm/site leakage check
    p("## 4. Farm-level split leakage check\n")
    leakers = [f for f, ss in farm_split.items() if len(ss) > 1]
    p(f"- Distinct farms (frm_id): **{len(farm_split)}**")
    p(f"- Farms appearing in BOTH train and val: **{len(leakers)}**")
    if leakers:
        p("\n  ⚠ Same farms exist in both splits — for proper generalization eval,")
        p("  consider grouping by `frm_id` when re-splitting.\n")
        for f in leakers[:10]:
            p(f"    - {f} → {sorted(farm_split[f])}")
    p("")

    # 5. site-level diversity
    p("## 5. Site (indvd_code) diversity per class (top 5)\n")
    for cls in classes:
        sites = site_by_class[cls]
        top = sites.most_common(5)
        s = ", ".join(f"{k}({v:,})" for k, v in top)
        p(f"- **{cls}** — distinct sites: {len(sites)};  top: {s}")
    p("")

    # 6. environment sensor ranges
    p("## 6. Environment sensors (sanity)\n")
    if env_inr_tp:
        p(f"- inner temperature : min={min(env_inr_tp):.1f}, median={median(env_inr_tp):.1f}, max={max(env_inr_tp):.1f}, σ={pstdev(env_inr_tp):.1f}")
    if env_inr_hd:
        p(f"- inner humidity    : min={min(env_inr_hd):.1f}, median={median(env_inr_hd):.1f}, max={max(env_inr_hd):.1f}, σ={pstdev(env_inr_hd):.1f}")
    p("")

    if bad:
        p(f"## 7. Parse errors: {len(bad)}\n")
        for jp, err in bad[:5]:
            p(f"  - {jp}: {err}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\n[done] report → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
