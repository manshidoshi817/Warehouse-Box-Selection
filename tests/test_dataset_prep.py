"""Tests for training/prepare_dataset.py using a synthetic Roboflow-style export."""
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from training.prepare_dataset import prepare


def build_export(tmp_path: Path) -> Path:
    root = tmp_path / "export"
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    for i in range(10):
        cv2.imwrite(str(root / "train" / "images" / f"img{i}.jpg"), np.zeros((32, 32, 3), np.uint8))
        lines = [
            "1 0.5 0.5 0.2 0.2",      # real box (class 1 = box)
            "1 0.1 0.1 0.01 0.01",    # tiny -> removed
            "0 0.3 0.3 0.2 0.2",      # stray class -> removed
        ]
        if i == 0:
            lines.append("1 0.5 0.5 0.1 0.1 0.3 0.1 0.3 0.4 0.5 0.4")  # polygon -> converted
        (root / "train" / "labels" / f"img{i}.txt").write_text("\n".join(lines) + "\n")
    (root / "data.yaml").write_text(yaml.safe_dump({"nc": 2, "names": ["Object-detection", "box"]}))

    archive = tmp_path / "export.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for p in root.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(root))
    return archive


def test_prepare_cleans_and_splits(tmp_path):
    stats = prepare(build_export(tmp_path), tmp_path / "out", min_area=0.003, val_fraction=0.2)
    assert stats["train_images"] + stats["valid_images"] == 10
    assert stats["valid_images"] == 2
    assert stats["kept_labels"] == 11  # 10 boxes + 1 polygon converted
    assert stats["removed_labels"] == 20  # 10 tiny + 10 stray

    cfg = yaml.safe_load(Path(stats["data_yaml"]).read_text())
    assert cfg["nc"] == 1 and cfg["names"] == ["box"]

    classes = set()
    for lbl in Path(cfg["path"]).rglob("labels/*.txt"):
        for line in lbl.read_text().splitlines():
            parts = line.split()
            assert len(parts) == 5
            classes.add(parts[0])
    assert classes == {"0"}


def test_prepare_max_boxes_drops_dense_images(tmp_path):
    stats = prepare(build_export(tmp_path), tmp_path / "out", max_boxes=1)
    assert "img0" in stats["dropped_images"]  # the only image with 2 boxes after cleaning
