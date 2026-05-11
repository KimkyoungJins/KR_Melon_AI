"""Dump calibration tensors as .npy files for MERA's quantizer.

MERA expects either a single .npy / .npz file or a directory of per-sample .npy
files (shape = input shape, e.g., 1×3×160×160 or 3×160×160).

Usage:
    python scripts/dump_calib_npy.py --out-dir docker_work/calib_npy --n 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import MelonDataset, build_transforms  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data/processed"))
    ap.add_argument("--out-dir", type=Path, default=Path("docker_work/calib_npy"))
    ap.add_argument("--n", type=int, default=100, help="number of calibration samples")
    ap.add_argument("--input-size", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    _, eval_tf = build_transforms(args.input_size)
    ds = MelonDataset(args.data_root / "metadata.csv", args.data_root, "train", eval_tf)
    gen = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, generator=gen)

    count = 0
    for x, _ in loader:
        # x shape: (1, 3, H, W). Save as numpy float32, shape (1, 3, H, W) (NCHW)
        arr = x.numpy().astype(np.float32)
        np.save(args.out_dir / f"{count:04d}.npy", arr)
        count += 1
        if count >= args.n:
            break

    print(f"[done] {count} calibration .npy files → {args.out_dir}")
    print(f"  shape per file = {arr.shape}, dtype = {arr.dtype}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
