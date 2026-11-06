"""Prepare a Roboflow export, train YOLO, and copy the best weights into models/best.pt.

Local / Colab usage:
    pip install ultralytics pyyaml
    python -m training.train --data my_roboflow_export.zip --epochs 100
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .prepare_dataset import prepare


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="Roboflow YOLO export (.zip or folder)")
    ap.add_argument("--workdir", default="work/dataset")
    ap.add_argument("--target-class", default="box")
    ap.add_argument("--min-area", type=float, default=0.003, help="drop labels smaller than this image fraction")
    ap.add_argument("--max-boxes", type=int, default=None, help="drop images with more labels than this")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--model", default="yolov8n.pt", help="pretrained weights to start from")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--name", default="box_detector")
    ap.add_argument("--export-to", default="models/best.pt")
    args = ap.parse_args()

    stats = prepare(args.data, args.workdir, args.target_class, args.min_area,
                    args.val_fraction, max_boxes=args.max_boxes)
    print(f"Dataset ready: {stats['train_images']} train / {stats['valid_images']} valid images, "
          f"{stats['kept_labels']} labels kept, {stats['removed_labels']} removed")
    if stats["dense_images"]:
        print("Warning: images with a very large number of labels (check them in Roboflow):",
              stats["dense_images"])

    from ultralytics import YOLO  # imported here so dataset prep works without it

    model = YOLO(args.model)
    model.train(
        data=stats["data_yaml"], epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, project="runs", name=args.name, exist_ok=True,
    )
    metrics = model.val()
    print(f"mAP50: {metrics.box.map50:.3f} | mAP50-95: {metrics.box.map:.3f}")

    best = Path(model.trainer.best)
    export = Path(args.export_to)
    export.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(best, export)
    print(f"Saved best weights to {export.resolve()}")


if __name__ == "__main__":
    main()
