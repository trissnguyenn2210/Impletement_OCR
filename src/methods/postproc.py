# -*- coding: utf-8 -*-
"""Chuyen output THO cua model -> canonical doc.

DIEM QUAN TRONG NHAT CUA CA REPO:
    Prediction phai di qua DUNG nhung buoc chuan hoa ma GROUND TRUTH da di qua.

Vi du: GT cho TIMESTAMP da bi strip tien to 'Ngay :' va tach ngay / gio
(RULE 3 trong data/gt_mcocr.py). Neu prediction KHONG strip, ta se so
    pred 'Ngay : 11/08/2020'   vs   gt '11/08/2020'
va ket luan model doc sai, trong khi model doc DUNG. Day la loi "benchmark
tu lam hong minh" - no khong lam chuong trinh crash nen rat kho phat hien.

Vi vay module nay import truc tiep TS_PREFIX / _is_amount tu gt_mcocr,
KHONG copy lai logic. Mot nguon su that duy nhat.
"""
import re

from data.gt_mcocr import RE_DATE, RE_TIME, TS_PREFIX, _is_amount

# nhan cua layout detector  ->  truong trong canonical schema
LABEL2FIELD = {
    "SELLER": ("header", "merchant"),
    "ADDRESS": ("header", "address"),
    "TIMESTAMP": ("header", "__timestamp"),   # xu ly rieng: tach ngay/gio
    "TOTAL_COST": ("payment", "total"),
    # D2 (v3) co thêm cac nhan nay; ITEM_* xu ly rieng o group_pred_items
    "PHONE": ("header", "phone"),
    "TOTAL": ("payment", "total"),
}
ITEM_LABELS = {"ITEM_NAME": "name", "ITEM_QTY": "quantity",
               "ITEM_PRICE": "unit_price", "ITEM_AMOUNT": "amount"}


def clean_timestamp(text: str) -> tuple[str | None, str | None]:
    """RULE 3 giong GT: bo tien to, tach ngay va gio."""
    t = TS_PREFIX.sub("", (text or "").strip())
    d = RE_DATE.search(t)
    h = RE_TIME.search(t)
    return (d.group(0).replace(" ", "") if d else None,
            h.group(0) if h else None)


def pick_total(texts: list[str]) -> str | None:
    """RULE 4 giong GT: trong cac vung TOTAL_COST, lay SO cuoi cung.

    Vung TOTAL_COST thuong gom ca chu khoa ('Tong tien:') va gia tri
    ('74,000'). Lay thang text dau tien se cho ra chu khoa -> diem 0 oan.
    """
    nums = [t for t in texts if _is_amount(t)]
    if nums:
        return nums[-1]
    return texts[-1] if texts else None


def regions_to_doc(regions: list[dict]) -> dict:
    """regions = [{'label','text','box':[x1,y1,x2,y2]}, ...] -> doc tho.

    Gom nhieu vung cung nhan: SELLER noi bang space, ADDRESS noi bang ', ',
    giong hệt GT. Thu tu theo y (tren xuong duoi) roi x (trai sang phai).
    """
    regions = sorted(regions, key=lambda r: (r["box"][1], r["box"][0]))
    bucket: dict[str, list[str]] = {}
    items_raw = []
    for r in regions:
        lab = (r.get("label") or "").upper()
        txt = (r.get("text") or "").strip()
        if not txt:
            continue
        if lab in ITEM_LABELS:
            items_raw.append(r)
        else:
            bucket.setdefault(lab, []).append(txt)

    header, payment = {}, {}
    for lab, texts in bucket.items():
        if lab not in LABEL2FIELD:
            continue
        sec, key = LABEL2FIELD[lab]
        if key == "__timestamp":
            d, h = clean_timestamp(" ".join(texts))
            header["date"], header["time"] = d, h
        elif key == "total":
            payment["total"] = pick_total(texts)
        elif key == "address":
            header["address"] = ", ".join(texts)
        elif key == "merchant":
            header["merchant"] = " ".join(texts)
        else:
            (header if sec == "header" else payment)[key] = texts[0]

    return {"header": header, "payment": payment,
            "line_items": group_pred_items(items_raw) if items_raw else None}


def group_pred_items(regions: list[dict], y_tol_ratio: float = 0.6) -> list[dict]:
    """Gom ITEM_* thanh tung mon theo KHOANG Y (giong data/gt_v3.py).

    KHONG gom theo chi so, vi so luong nhan moi loai KHONG bang nhau:
    nhieu template khong in don gia -> ITEM_PRICE it hon ITEM_NAME.
    Gom theo chi so se lech toan bo bang.
    """
    if not regions:
        return []
    heights = [r["box"][3] - r["box"][1] for r in regions] or [1]
    tol = max(1.0, (sum(heights) / len(heights)) * y_tol_ratio)

    rows: list[list[dict]] = []
    for r in sorted(regions, key=lambda r: (r["box"][1] + r["box"][3]) / 2):
        cy = (r["box"][1] + r["box"][3]) / 2
        if rows and abs(cy - rows[-1][0]["_cy"]) <= tol:
            r["_cy"] = rows[-1][0]["_cy"]
            rows[-1].append(r)
        else:
            r["_cy"] = cy
            rows.append([r])

    items = []
    for row in rows:
        it = {"name": None, "unit": None, "quantity": None,
              "unit_price": None, "amount": None}
        for r in sorted(row, key=lambda r: r["box"][0]):
            key = ITEM_LABELS.get((r.get("label") or "").upper())
            if key and it[key] is None:
                it[key] = (r.get("text") or "").strip() or None
        if any(it.values()):
            items.append(it)
    return items


# ------------------------------------------------------- output cua VLM / LLM
_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


def parse_json_loose(text: str) -> dict | None:
    """Doc JSON tu output VLM/LLM. Tra None neu that bai -> tinh vao parse_fail.

    VLM rat hay boc JSON trong ```json ... ``` hoac them mot cau dan nhap.
    Ba lan thu, tu nghiem den long. Tra None chu KHONG tra {} - vi {} se bi
    tinh la "model tra loi rong" thay vi "model tra loi khong doc duoc".
    """
    import json

    if not text:
        return None
    for cand in (text,
                 (_FENCE.search(text).group(1) if _FENCE.search(text) else None),
                 _slice_braces(text)):
        if not cand:
            continue
        try:
            d = json.loads(cand)
            if isinstance(d, dict):
                return d
        except Exception:
            continue
    return None


def _slice_braces(text: str) -> str | None:
    i, j = text.find("{"), text.rfind("}")
    return text[i:j + 1] if 0 <= i < j else None
