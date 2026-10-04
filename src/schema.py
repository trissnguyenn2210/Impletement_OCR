# -*- coding: utf-8 -*-
"""Canonical schema. MOI phuong phap (A, B, C, D) phai tra ve dung dang nay.

Quy uoc quan trong:
    truong = None   -> KHONG co tren hoa don (dung de do Hallucination rate)
    line_items = None -> bo data KHONG CO NHAN item  -> metric tra N/A
    line_items = []   -> CO nhan, va hoa don khong co mon nao -> metric tra 0
"""
from cfg import HEADER_FIELDS, PAYMENT_FIELDS, ITEM_FIELDS


def empty_doc(img_id: str = "", quality=None) -> dict:
    return {
        "img_id": img_id,
        "quality": quality,
        "header": {k: None for k in HEADER_FIELDS},
        "line_items": None,
        "payment": {k: None for k in PAYMENT_FIELDS},
    }


def empty_item() -> dict:
    return {k: None for k in ITEM_FIELDS}


def normalize_doc(d: dict | None, img_id: str = "", quality=None) -> dict:
    """Bo sung truong thieu, loai truong la. Dung cho CA gt VA prediction."""
    out = empty_doc(img_id, quality)
    if not isinstance(d, dict):
        return out
    out["img_id"] = d.get("img_id") or img_id
    out["quality"] = d.get("quality", quality)
    for block, keys in (("header", HEADER_FIELDS), ("payment", PAYMENT_FIELDS)):
        src = d.get(block) or {}
        if isinstance(src, dict):
            for k in keys:
                v = src.get(k)
                out[block][k] = None if v in ("", None) else str(v)
    items = d.get("line_items")
    if items is None:
        out["line_items"] = None
    elif isinstance(items, list):
        fixed = []
        for it in items:
            e = empty_item()
            if isinstance(it, dict):
                for k in ITEM_FIELDS:
                    v = it.get(k)
                    e[k] = None if v in ("", None) else str(v)
            fixed.append(e)
        out["line_items"] = fixed
    else:
        out["line_items"] = None
    return out


def validate_doc(d: dict) -> list[str]:
    """Tra ve danh sach loi. Rong = hop le."""
    errs = []
    if not isinstance(d, dict):
        return ["khong phai dict"]
    for block, keys in (("header", HEADER_FIELDS), ("payment", PAYMENT_FIELDS)):
        if not isinstance(d.get(block), dict):
            errs.append(f"thieu block {block}")
            continue
        extra = set(d[block]) - set(keys)
        if extra:
            errs.append(f"{block} co truong la: {sorted(extra)}")
    it = d.get("line_items")
    if it is not None and not isinstance(it, list):
        errs.append("line_items phai la None hoac list")
    if isinstance(it, list):
        for i, e in enumerate(it):
            if not isinstance(e, dict):
                errs.append(f"line_items[{i}] khong phai dict")
            else:
                extra = set(e) - set(ITEM_FIELDS)
                if extra:
                    errs.append(f"line_items[{i}] co truong la: {sorted(extra)}")
    return errs
