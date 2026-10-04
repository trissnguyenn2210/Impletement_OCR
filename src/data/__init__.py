# -*- coding: utf-8 -*-
"""Nap ground truth + giai duong dan anh cho tung tap test."""
import json
from pathlib import Path

import cfg
from . import splits


def gt_file(test_set: str) -> Path:
    ds, sp = cfg.TEST_SETS[test_set]
    return cfg.GT_DIR / f"{ds}_{sp}.json"


def load_gt(test_set: str) -> dict[str, dict]:
    f = gt_file(test_set)
    if not f.exists():
        raise FileNotFoundError(f"chua co {f}. Chay truoc: python build_gt.py")
    return json.loads(f.read_text(encoding="utf-8"))


def image_path(test_set: str, stem: str) -> Path:
    """Giai duong dan anh. D1 dung .jpg, D2 dung .png (co fallback)."""
    ds, _ = cfg.TEST_SETS[test_set]
    if ds == "d1":
        p = cfg.D1_IMAGES / f"{stem}.jpg"
        if p.exists():
            return p
        for ext in (".png", ".jpeg", ".JPG"):
            q = cfg.D1_IMAGES / f"{stem}{ext}"
            if q.exists():
                return q
        return p                                    # tra duong dan du khong ton tai
    for sp in ("train", "val"):                     # D2: thu ca hai thu muc
        for ext in (".png", ".jpg", ".jpeg"):
            q = cfg.D2 / sp / "images" / f"{stem}{ext}"
            if q.exists():
                return q
    return cfg.D2 / "train" / "images" / f"{stem}.png"


__all__ = ["splits", "load_gt", "gt_file", "image_path"]
