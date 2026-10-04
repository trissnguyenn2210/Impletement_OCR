# -*- coding: utf-8 -*-
"""Diem cho cac truong header + payment, va Hallucination rate."""
from .core import anls, numeric_match, norm_date_em, exact, nfc

# Moi truong dung DUNG MOT loai metric
FIELD_METRIC = {
    # text dai -> ANLS
    "merchant": anls, "address": anls,
    # ngay/gio -> chuan hoa roi so bang
    "date": norm_date_em, "time": norm_date_em,
    # tien va so -> so GIA TRI
    "total": numeric_match, "discount": numeric_match, "vat": numeric_match,
    "tendered": numeric_match, "change": numeric_match,
    # ma -> exact
    "invoice_no": exact, "tax_code": exact, "phone": exact,
    "counter": exact, "cashier": exact,
}


def score_doc(pred: dict | None, gt: dict) -> dict:
    """Diem tung truong cua MOT anh.

    Truong co gt = None  -> KHONG tinh diem, chuyen sang do hallucination.
    Nen MAU SO moi truong KHAC NHAU tren toan tap test.
    """
    per, hn, hd = {}, 0, 0
    for block in ("header", "payment"):
        for k, g in (gt.get(block) or {}).items():
            p = (pred.get(block) or {}).get(k) if isinstance(pred, dict) else None
            if g is None or nfc(g) == "":
                hd += 1
                hn += int(nfc(p) != "")          # model tra gia tri cho truong rong = bia
                continue
            per[k] = FIELD_METRIC.get(k, anls)(p, g)
    return {"img_id": gt.get("img_id"), "quality": gt.get("quality"),
            "per_field": per, "halluc_n": hn, "halluc_d": hd}


def aggregate(docs: list[dict]) -> dict:
    """Cong don qua nhieu anh. Tra ve % (0-100)."""
    acc, cnt, hn, hd = {}, {}, 0, 0
    for d in docs:
        for k, v in d["per_field"].items():
            acc[k] = acc.get(k, 0.0) + v
            cnt[k] = cnt.get(k, 0) + 1
        hn += d["halluc_n"]
        hd += d["halluc_d"]
    per = {k: 100.0 * acc[k] / cnt[k] for k in sorted(acc)}
    return {
        "MFS": (sum(per.values()) / len(per)) if per else None,
        "per_field": per,
        "n_per_field": {k: cnt[k] for k in sorted(cnt)},   # PHAI in ra bang
        "halluc": (100.0 * hn / hd) if hd else None,
        "halluc_n": hn, "halluc_d": hd,
        "n_docs": len(docs),
    }
