"""Phase 2: train MobileNetV2-0.35 on the prepared 5-class melon dataset.

- Inputs : data/processed/{train,val,test}/<class>/*.jpg via metadata.csv
- Output : models/melon_classifier_fp32.pth (best val acc)
           models/training_log.csv
           models/training_summary.json
- TensorBoard logs in runs/

Designed for M1/M2 (MPS) but auto-detects CUDA / falls back to CPU.

Usage:
    python scripts/train.py
    python scripts/train.py --epochs 50 --batch-size 96 --lr 2e-3
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import CLASS_TO_IDX, CLASSES  # noqa: E402

# ImageNet normalization — convention; even from-scratch training works fine with these stats.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class MelonDataset(Dataset):
    def __init__(self, csv_path: Path, root: Path, split: str, transform):
        with csv_path.open(encoding="utf-8") as f:
            self.rows = [r for r in csv.DictReader(f) if r["split"] == split]
        self.root = root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        r = self.rows[idx]
        img = Image.open(self.root / r["image_path"]).convert("RGB")
        return self.transform(img), CLASS_TO_IDX[r["class_name"]]


def build_transforms(input_size: int):
    """Strong color augmentation given the day/night class confound we found in analysis."""
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(input_size),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])
    return train_tf, eval_tf


def build_model(num_classes: int, width_mult: float = 1.0, pretrained: bool = True) -> nn.Module:
    """MobileNetV2 with optional ImageNet-pretrained backbone.

    Activations are ReLU6 (Ethos-U55 op whitelist). Pretrained weights only
    available at width=1.0 from torchvision; for other widths we train from scratch.
    """
    if pretrained and width_mult == 1.0:
        model = mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    else:
        model = mobilenet_v2(weights=None, width_mult=width_mult)

    in_feat = model.classifier[1].in_features
    # Re-init classifier head for our 5 classes (head must be trained from scratch).
    model.classifier[1] = nn.Linear(in_feat, num_classes)
    return model


def class_weights(rows: list[dict], num_classes: int) -> torch.Tensor:
    counts = Counter(CLASS_TO_IDX[r["class_name"]] for r in rows)
    total = sum(counts.values())
    w = torch.ones(num_classes)
    for k, v in counts.items():
        w[k] = total / (num_classes * v)
    return w


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total = correct = 0
    loss_sum = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum / total, correct / total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("models"))
    ap.add_argument("--input-size", type=int, default=160)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--width-mult", type=float, default=1.0)
    ap.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=True,
                    help="Load ImageNet pretrained weights (only at width-mult=1.0).")
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"[device] {device}")

    csv_path = args.data_root / "metadata.csv"
    train_tf, eval_tf = build_transforms(args.input_size)
    train_ds = MelonDataset(csv_path, args.data_root, "train", train_tf)
    val_ds = MelonDataset(csv_path, args.data_root, "val", eval_tf)
    test_ds = MelonDataset(csv_path, args.data_root, "test", eval_tf)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)}")

    pin = device.type == "cuda"
    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=pin, drop_last=False)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=pin)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=pin)

    model = build_model(num_classes=len(CLASSES), width_mult=args.width_mult,
                        pretrained=args.pretrained).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    pre = "pretrained" if (args.pretrained and args.width_mult == 1.0) else "from-scratch"
    print(f"[model] MobileNetV2 width_mult={args.width_mult} ({pre})  params={n_params:,}  "
          f"({n_params * 4 / 1e6:.2f} MB fp32 / ~{n_params / 1e6:.2f} MB int8)")

    weights = class_weights(train_ds.rows, len(CLASSES))
    print(f"[loss] class weights = {[round(w, 3) for w in weights.tolist()]}")
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    run_id = time.strftime("mnv2_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=f"runs/{run_id}")
    print(f"[tb] runs/{run_id}")

    log_rows: list[dict] = []
    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0
    ckpt_path = args.out_dir / "melon_classifier_fp32.pth"

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        dt = time.time() - t0
        cur_lr = scheduler.get_last_lr()[0]

        writer.add_scalars("loss", {"train": tr_loss, "val": va_loss}, epoch)
        writer.add_scalars("acc", {"train": tr_acc, "val": va_acc}, epoch)
        writer.add_scalar("lr", cur_lr, epoch)
        log_rows.append({
            "epoch": epoch, "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc, "lr": cur_lr, "time_s": dt,
        })

        improved = va_acc > best_val_acc
        marker = "*" if improved else " "
        print(f"  {marker} ep{epoch:>2}/{args.epochs}  "
              f"train loss={tr_loss:.4f} acc={tr_acc:.4f} | "
              f"val loss={va_loss:.4f} acc={va_acc:.4f} | "
              f"lr={cur_lr:.2e}  ({dt:.1f}s)")

        if improved:
            best_val_acc = va_acc
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "model_state": model.state_dict(),
                "epoch": epoch, "val_acc": va_acc,
                "args": vars(args), "classes": list(CLASSES),
            }, ckpt_path)
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"[earlystop] no improvement in {args.patience} epochs; stopping.")
                break

    # Save training log
    with (args.out_dir / "training_log.csv").open("w", newline="", encoding="utf-8") as f:
        if log_rows:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader()
            w.writerows(log_rows)

    # Final test eval with best checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    te_loss, te_acc = evaluate(model, test_loader, criterion, device)
    print(f"\n[best] epoch={best_epoch}  val_acc={best_val_acc:.4f}")
    print(f"[test] loss={te_loss:.4f}  acc={te_acc:.4f}")

    summary = {
        "run_id": run_id,
        "device": str(device),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "test_acc": te_acc,
        "test_loss": te_loss,
        "params": n_params,
        "model_size_fp32_mb": round(n_params * 4 / 1e6, 3),
        "input_size": args.input_size,
        "width_mult": args.width_mult,
        "epochs_run": len(log_rows),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    }
    (args.out_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[done] {ckpt_path}")
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
