"""Evaluate a trained checkpoint on test (default) split.

Outputs:
- terminal: per-class precision/recall/F1 + confusion matrix table
- models/confusion_matrix_<split>.png
- models/eval_<split>.json

Usage:
    python scripts/eval_model.py
    python scripts/eval_model.py --split val --ckpt models/melon_classifier_fp32.pth
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLASSES  # noqa: E402
from train import MelonDataset, build_model, build_transforms, pick_device  # noqa: E402


def plot_confusion(cm: np.ndarray, classes, out_path: Path, split: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    for fp in (
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
    ):
        if Path(fp).exists():
            font_manager.fontManager.addfont(fp)
            plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
            break

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(7, 5.8))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes))); ax.set_xticklabels(classes, rotation=15)
    ax.set_yticks(range(len(classes))); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title(f"Confusion matrix — {split}")
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n({cm_norm[i, j]:.2f})",
                    ha="center", va="center", fontsize=8,
                    color="white" if cm_norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"[plot] {out_path}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, default=Path("models/melon_classifier_fp32.pth"))
    ap.add_argument("--data-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=2)
    args = ap.parse_args()

    device = pick_device()
    print(f"[device] {device}")
    if not args.ckpt.exists():
        print(f"[fail] checkpoint not found: {args.ckpt}", file=sys.stderr)
        return 2

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    width_mult = ckpt["args"].get("width_mult", 0.35)
    input_size = ckpt["args"].get("input_size", 160)

    _, eval_tf = build_transforms(input_size)
    ds = MelonDataset(args.data_root / "metadata.csv", args.data_root, args.split, eval_tf)
    loader = DataLoader(ds, args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model(num_classes=len(CLASSES), width_mult=width_mult).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    all_true: list[int] = []
    all_pred: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred = model(x).argmax(1).cpu().numpy()
            all_pred.extend(pred.tolist())
            all_true.extend(y.numpy().tolist())

    cm = confusion_matrix(all_true, all_pred, labels=list(range(len(CLASSES))))
    acc = float(np.mean(np.array(all_true) == np.array(all_pred)))
    report = classification_report(
        all_true, all_pred, target_names=list(CLASSES), digits=4, zero_division=0
    )

    print(f"\n[{args.split}] N={len(ds)}  accuracy={acc:.4f}")
    print(report)
    print("Confusion matrix (rows=true, cols=pred):")
    print(" " * 12 + "".join(f"{c[:8]:>10}" for c in CLASSES))
    for i, row in enumerate(cm):
        print(f"  {CLASSES[i][:8]:<10}" + "".join(f"{v:>10}" for v in row))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_confusion(cm, list(CLASSES), args.out_dir / f"confusion_matrix_{args.split}.png", args.split)
    (args.out_dir / f"eval_{args.split}.json").write_text(
        json.dumps({
            "split": args.split, "accuracy": acc, "n_samples": len(ds),
            "ckpt_epoch": ckpt.get("epoch"), "ckpt_val_acc": ckpt.get("val_acc"),
            "confusion_matrix": cm.tolist(), "classes": list(CLASSES),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"[done] eval saved → models/eval_{args.split}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
