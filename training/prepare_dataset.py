"""Turn a Roboflow / YOLO export into a clean, single-class ("box") training dataset.

What it does
  * keeps only the target class and renumbers it to 0 (drops stray classes)
  * converts polygon labels to boxes (if the export was segmentation)
  * removes tiny boxes (default: under 0.3% of the image area) that are usually noise
  * creates a validation split if the export has none
  * rewrites data.yaml with absolute paths

Usage:  python -m training.prepare_dataset --data export.zip --out work/dataset
"""
from __future__ import annotations

import argparse
import random
import shutil
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import yaml

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DENSE_WARNING = 60  # warn when one image has more labels than this


def _load_names(data_yaml: Path) -> List[str]:
    cfg = yaml.safe_load(data_yaml.read_text())
    names = cfg.get("names")
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=int)]
    return list(names or [])


def _find_root(extracted: Path) -> Path:
    if (extracted / "data.yaml").exists():
        return extracted
    for candidate in extracted.rglob("data.yaml"):
        return candidate.parent
    raise FileNotFoundError("data.yaml was not found in the dataset.")


def _convert_line(parts: List[str], target_idx: int, min_area: float) -> Optional[str]:
    """Return a normalized 'cls x y w h' line for the target class, or None to drop it."""
    try:
        cls = int(float(parts[0]))
        vals = [float(v) for v in parts[1:]]
    except ValueError:
        return None
    if cls != target_idx:
        return None
    if len(vals) == 4:
        x, y, w, h = vals
    elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon -> bounding box
        xs, ys = vals[0::2], vals[1::2]
        x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
        x, y, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
    else:
        return None
    if w <= 0 or h <= 0 or w * h < min_area:
        return None
    return f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"


def _image_for(label: Path, images_dir: Path) -> Optional[Path]:
    for ext in IMG_EXT | {e.upper() for e in IMG_EXT}:
        candidate = images_dir / f"{label.stem}{ext}"
        if candidate.exists():
            return candidate
    return None


def prepare(
    source: str | Path,
    out_dir: str | Path,
    target_class: str = "box",
    min_area: float = 0.003,
    val_fraction: float = 0.2,
    seed: int = 42,
    max_boxes: Optional[int] = None,
) -> Dict:
    source, out_dir = Path(source), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    if source.is_file() and source.suffix.lower() == ".zip":
        with zipfile.ZipFile(source) as zf:
            zf.extractall(out_dir)
    elif source.is_dir():
        shutil.copytree(source, out_dir, dirs_exist_ok=True)
    else:
        raise FileNotFoundError(f"Dataset not found: {source}")

    root = _find_root(out_dir)
    names = _load_names(root / "data.yaml")
    if target_class in names:
        target_idx = names.index(target_class)
    elif len(names) == 1:
        target_idx = 0
    else:
        raise ValueError(f"Class '{target_class}' not found. Classes in the dataset: {names}")

    if (root / "val").exists() and not (root / "valid").exists():
        (root / "val").rename(root / "valid")

    stats: Dict = {"kept_labels": 0, "removed_labels": 0, "dropped_images": [], "dense_images": []}

    for split in ("train", "valid", "test"):
        labels_dir, images_dir = root / split / "labels", root / split / "images"
        if not labels_dir.exists():
            continue
        for label in sorted(labels_dir.glob("*.txt")):
            kept = []
            for line in label.read_text().splitlines():
                parts = line.split()
                if not parts:
                    continue
                converted = _convert_line(parts, target_idx, min_area)
                if converted:
                    kept.append(converted)
                else:
                    stats["removed_labels"] += 1

            if max_boxes is not None and len(kept) > max_boxes:
                image = _image_for(label, images_dir)
                if image:
                    image.unlink()
                label.unlink()
                stats["dropped_images"].append(label.stem)
                continue

            if len(kept) > DENSE_WARNING:
                stats["dense_images"].append((label.stem, len(kept)))
            label.write_text("\n".join(kept) + ("\n" if kept else ""))
            stats["kept_labels"] += len(kept)

    # Make sure there is a validation split.
    valid_images = root / "valid" / "images"
    has_valid = valid_images.exists() and any(p.suffix.lower() in IMG_EXT for p in valid_images.iterdir())
    if not has_valid:
        train_images = sorted(
            p for p in (root / "train" / "images").iterdir() if p.suffix.lower() in IMG_EXT
        )
        if len(train_images) < 2:
            raise ValueError("Need at least 2 images to create a train/validation split.")
        random.Random(seed).shuffle(train_images)
        n_val = min(max(1, int(len(train_images) * val_fraction)), len(train_images) - 1)
        (root / "valid" / "images").mkdir(parents=True, exist_ok=True)
        (root / "valid" / "labels").mkdir(parents=True, exist_ok=True)
        for image in train_images[:n_val]:
            label = root / "train" / "labels" / f"{image.stem}.txt"
            shutil.move(str(image), root / "valid" / "images" / image.name)
            if label.exists():
                shutil.move(str(label), root / "valid" / "labels" / label.name)

    for cache in root.rglob("*.cache"):
        cache.unlink()

    cfg = {
        "path": str(root.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "nc": 1,
        "names": ["box"],
    }
    test_images = root / "test" / "images"
    if test_images.exists() and any(test_images.iterdir()):
        cfg["test"] = "test/images"
    (root / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    def count(split: str) -> int:
        d = root / split / "images"
        return sum(1 for p in d.iterdir() if p.suffix.lower() in IMG_EXT) if d.exists() else 0

    stats.update(
        data_yaml=str((root / "data.yaml").resolve()),
        train_images=count("train"),
        valid_images=count("valid"),
        test_images=count("test"),
    )
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Roboflow YOLO export (.zip or folder)")
    ap.add_argument("--out", default="work/dataset")
    ap.add_argument("--target-class", default="box")
    ap.add_argument("--min-area", type=float, default=0.003)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--max-boxes", type=int, default=None, help="drop images with more labels than this")
    args = ap.parse_args()
    stats = prepare(args.data, args.out, args.target_class, args.min_area, args.val_fraction,
                    max_boxes=args.max_boxes)
    for key, value in stats.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
