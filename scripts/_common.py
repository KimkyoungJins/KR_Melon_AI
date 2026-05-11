"""Shared helpers for dataset scripts.

Layout assumed:
  <aihub_root>/Training/01.원천데이터/TS_<class>.zip      (image zips, NOT extracted)
  <aihub_root>/Training/02.라벨링데이터/TL_<class>.zip
  <aihub_root>/Validation/01.원천데이터/VS_<class>.zip
  <aihub_root>/Validation/02.라벨링데이터/VL_<class>.zip

  <labels_root>/train/<class>/*.json   (extracted labels — produced before)
  <labels_root>/val/<class>/*.json
"""
from __future__ import annotations

import io
import json
import unicodedata
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

CLASSES: tuple[str, ...] = (
    "노균병",
    "노균병유사",
    "흰가루병",
    "흰가루병유사",
    "정상",
)
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASSES)}

# Default AIHub root inside this project. Override with --aihub-root if you moved it.
DEFAULT_AIHUB_ROOT = Path("data/247.지능형 스마트팜(참외) 데이터/01-1.정식개방데이터")


@dataclass(frozen=True)
class Sample:
    """A labeled image, with enough metadata to fetch its bytes from a zip later."""
    image_name: str          # e.g. "S005-FM01-005-2022-07-19-000021.jpg"
    class_name: str          # one of CLASSES
    farm_id: str             # frm_id, e.g. "FM01" — used for group-aware split
    site_id: str             # indvd_code, e.g. "S005"
    day_section: str         # "주간" / "야간"
    aihub_split: str         # "train" or "val" — original AIHub split (= which zip)


def normalize(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def get_image_zip(aihub_root: Path, aihub_split: str, class_name: str) -> Path:
    """Locate the image zip for a given (aihub_split, class)."""
    if aihub_split == "train":
        sub, prefix = "Training", "TS"
    elif aihub_split == "val":
        sub, prefix = "Validation", "VS"
    else:
        raise ValueError(f"unknown aihub_split: {aihub_split}")
    return aihub_root / sub / "01.원천데이터" / f"{prefix}_{class_name}.zip"


def _load_json(p: Path) -> dict | None:
    try:
        with p.open(encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def discover_from_labels(labels_root: Path) -> tuple[list[Sample], list[tuple[Path, str]]]:
    """Walk extracted labels/{train,val}/{class}/*.json → Sample list."""
    samples: list[Sample] = []
    errors: list[tuple[Path, str]] = []

    for split in ("train", "val"):
        split_dir = labels_root / split
        if not split_dir.is_dir():
            errors.append((split_dir, "missing split dir"))
            continue
        for cls in CLASSES:
            cls_dir = split_dir / cls
            if not cls_dir.is_dir():
                errors.append((cls_dir, "missing class dir"))
                continue
            for jp in cls_dir.rglob("*.json"):
                d = _load_json(jp)
                if d is None:
                    errors.append((jp, "json parse failed"))
                    continue
                base = d.get("base_info") or {}
                img = d.get("image") or {}
                corps = d.get("corps_info") or {}
                image_name = base.get("image_name")
                cn = normalize(img.get("class_name") or "")
                if not image_name or cn != cls:
                    errors.append((jp, f"image_name missing or class mismatch ({cn!r} vs dir {cls!r})"))
                    continue
                samples.append(Sample(
                    image_name=image_name,
                    class_name=cls,
                    farm_id=corps.get("frm_id", "?"),
                    site_id=corps.get("indvd_code", "?"),
                    day_section=img.get("day_section", "?"),
                    aihub_split=split,
                ))
    return samples, errors


def open_zip_for_class(aihub_root: Path, aihub_split: str, class_name: str) -> zipfile.ZipFile:
    """Open the image zip for a class. Caller is responsible for closing."""
    return zipfile.ZipFile(get_image_zip(aihub_root, aihub_split, class_name))


def read_image_bytes(zf: zipfile.ZipFile, image_name: str) -> bytes:
    """Read raw bytes for an image. AIHub zips store entries with a leading slash."""
    # Try with leading slash first (AIHub convention), fall back to plain.
    for candidate in (f"/{image_name}", image_name):
        try:
            return zf.read(candidate)
        except KeyError:
            continue
    raise KeyError(image_name)


def letterbox(img, target: int, fill: tuple[int, int, int] = (114, 114, 114)):
    """Aspect-preserving resize + pad to (target, target). PIL Image in/out."""
    from PIL import Image

    w, h = img.size
    scale = target / max(w, h)
    new_w, new_h = max(1, round(w * scale)), max(1, round(h * scale))
    img = img.resize((new_w, new_h), Image.BILINEAR)
    canvas = Image.new("RGB", (target, target), fill)
    canvas.paste(img, ((target - new_w) // 2, (target - new_h) // 2))
    return canvas


def open_image_from_bytes(b: bytes):
    """Open image bytes lazily (returns PIL Image; caller should convert/use within scope)."""
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    return Image.open(io.BytesIO(b))
