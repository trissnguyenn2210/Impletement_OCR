# -*- coding: utf-8 -*-
"""D1 (MC-OCR) -> canonical ground truth.

BON RULE, suy ra tu viec doc that 1.155 dong cua mcocr_train_df.csv:

  RULE 1  SELLER     8% anh co 2-3 vung        -> join bang dau cach
  RULE 2  ADDRESS    42% anh co >=2 vung (toi 5) -> join bang ", "
  RULE 3  TIMESTAMP  kem tien to khoa          -> strip ('Ngay:' 404 lan,
                     'Ngay ban:' 305, 'Thoi gian:' 119, ...) roi tach ngay/gio
  RULE 4  TOTAL_COST 84% anh co >=2 vung, LAN chu khoa voi gia tri
                     ('Tong tien:' + '74,000')  -> CHI lay vung co so
  RULE 5  loai anh co anno_num > 12 (2 anh nhan loi: 18 va 25 vung TOTAL_COST)

D1 KHONG co nhan ITEM_* -> line_items = None (khong phai []).
"""
import csv
import re

import cfg
from schema import normalize_doc

csv.field_size_limit(10_000_000)

TS_PREFIX = re.compile(
    r"^(Ngày bán|Ngày|Thời gian|NGÀY|Ngay|Giờ vào|Giờ ra|GIỜ|Giờ)\s*:?\s*", re.I)
RE_DATE = re.compile(r"\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}")
RE_TIME = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")
MAX_ANNO_NUM = 12


def _is_amount(t: str) -> bool:
    """Vung TOTAL_COST nao la GIA TRI chu khong phai chu khoa."""
    digits = re.sub(r"[^\d]", "", t)
    letters = re.sub(r"[\d.,\s]", "", t)
    return len(digits) >= 4 and len(letters) <= 2


def row_to_doc(row: dict) -> dict | None:
    """Tra ve canonical doc, hoac None neu dong bi loai."""
    try:
        if int(float(row.get("anno_num") or 0)) > MAX_ANNO_NUM:   # RULE 5
            return None
    except (TypeError, ValueError):
        return None

    texts = [t.strip() for t in (row.get("anno_texts") or "").split("|||")]
    labs = [l.strip() for l in (row.get("anno_labels") or "").split("|||")]
    if len(texts) != len(labs):
        return None

    def grab(label):
        return [t for t, l in zip(texts, labs) if l == label and t]

    seller = " ".join(grab("SELLER")).strip() or None                  # RULE 1
    address = ", ".join(grab("ADDRESS")).strip() or None               # RULE 2

    ts = " ".join(TS_PREFIX.sub("", t).strip() for t in grab("TIMESTAMP"))  # RULE 3
    md, mt = RE_DATE.search(ts), RE_TIME.search(ts)

    # RULE 4 - nhan goc co ca TOTAL_COST va typo TOTAL_TOTAL_COST
    cands = grab("TOTAL_COST") + grab("TOTAL_TOTAL_COST")
    nums = [t for t in cands if _is_amount(t)]

    try:
        q = float(row.get("anno_image_quality"))
    except (TypeError, ValueError):
        q = None

    return normalize_doc({
        "img_id": row["img_id"],
        "quality": q,
        "header": {"merchant": seller, "address": address,
                   "date": md.group(0).replace(" ", "") if md else None,
                   "time": mt.group(0) if mt else None},
        "line_items": None,            # D1 KHONG co nhan item
        "payment": {"total": nums[-1] if nums else None},
    }, img_id=row["img_id"], quality=q)


def load_all() -> tuple[dict[str, dict], dict]:
    """Tra ve ({stem: doc}, bao_cao)."""
    with open(cfg.D1_CSV, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    docs, dropped = {}, []
    for r in rows:
        doc = row_to_doc(r)
        if doc is None:
            dropped.append(r.get("img_id"))
            continue
        stem = re.sub(r"\.(jpg|png|jpeg)$", "", doc["img_id"], flags=re.I)
        docs[stem] = doc

    null_cnt = {}
    for d in docs.values():
        for block in ("header", "payment"):
            for k, v in d[block].items():
                if v is None:
                    null_cnt[k] = null_cnt.get(k, 0) + 1
    return docs, {"rows_read": len(rows), "docs_ok": len(docs),
                  "dropped": dropped, "null_per_field": null_cnt}
