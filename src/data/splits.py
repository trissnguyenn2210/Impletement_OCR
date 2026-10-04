# -*- coding: utf-8 -*-
"""Split cua D1 lay tu CHINH hai file text_recognition_*.txt co san trong dataset.

Ly do KHONG tu chia lai:
    Hai file do da chia theo ANH (da verify: 922 / 231, giao nhau = 0).
    Model recognition duoc train tren crop cua 922 anh do. Neu tu chia lai bang
    seed moi, mot so anh trong val MOI se co crop nam trong tap train cu
    -> recognition DA THAY text cua anh do -> diem cua A bi thoi phong.
"""
import re
from pathlib import Path

import cfg


def _imgs_from_rec_file(p: Path) -> set[str]:
    """mcocr_public_145013snoxg_3.jpg  ->  mcocr_public_145013snoxg"""
    out = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        crop = line.split("\t")[0]
        out.add(re.sub(r"_\d+\.(jpg|png|jpeg)$", "", crop, flags=re.I))
    return out


def d1_splits() -> dict[str, list[str]]:
    tr = _imgs_from_rec_file(cfg.D1_REC_TRAIN)
    va = _imgs_from_rec_file(cfg.D1_REC_VAL)
    overlap = tr & va
    if overlap:
        raise RuntimeError(
            f"RO RI SPLIT: {len(overlap)} anh nam o ca train va val, "
            f"vi du {sorted(overlap)[:3]}")
    return {"train": sorted(tr), "val": sorted(va), "all": sorted(tr | va)}


def d2_splits() -> dict[str, list[str]]:
    """D2 chia theo THU MUC train/ va val/."""
    out = {}
    for sp in ("train", "val"):
        d = cfg.D2 / sp / "labels"
        out[sp] = sorted(p.stem for p in d.glob("*.json"))
    out["all"] = sorted(out["train"] + out["val"])
    return out


def write_split_files() -> dict[str, int]:
    counts = {}
    for name, spl in (("d1", d1_splits()), ("d2", d2_splits())):
        for sp, ids in spl.items():
            f = cfg.SPLIT_DIR / f"{name}_{sp}.txt"
            f.write_text("\n".join(ids) + "\n", encoding="utf-8")
            counts[f"{name}_{sp}"] = len(ids)
    return counts


def load_split(test_set: str) -> list[str]:
    if test_set not in cfg.TEST_SETS:
        raise KeyError(f"tap test khong hop le: {test_set!r}. "
                       f"Hop le: {sorted(cfg.TEST_SETS)}")
    ds, sp = cfg.TEST_SETS[test_set]
    f = cfg.SPLIT_DIR / f"{ds}_{sp}.txt"
    if not f.exists():
        raise FileNotFoundError(f"chua co {f}. Chay: python build_gt.py")
    return [l for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
