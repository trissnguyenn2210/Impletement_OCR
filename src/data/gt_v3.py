# -*- coding: utf-8 -*-
"""D2 (vietnamese-receipts-v3) -> canonical ground truth.

Format nguon: mot file JSON moi anh
    {"file_name", "image_width", "image_height",
     "annotations": [{"label", "text", "box": [x1,y1,x2,y2]}, ...]}

8 nhan: SELLER ADDRESS TIMESTAMP TOTAL_COST ITEM_NAME ITEM_QTY ITEM_PRICE ITEM_AMOUNT

GOM LINE ITEMS THEO KHOANG Y, khong theo chi so.
  Ly do: so luong nhan KHONG bang nhau (da kiem tra tren 4.995 anh)
      ITEM_NAME   18.565
      ITEM_AMOUNT 18.565
      ITEM_QTY    17.245
      ITEM_PRICE  13.159     <- thieu 5.406 so voi ITEM_NAME
  Neu ghep theo chi so i thi cac hoa don thieu ITEM_PRICE se bi LECH MON.
"""
import json
import re
from pathlib import Path

import cfg
from schema import normalize_doc

RE_DATE = re.compile(r"\d{1,2}\s*[/\-.]\s*\d{1,2}\s*[/\-.]\s*\d{2,4}")
RE_TIME = re.compile(r"\d{1,2}:\d{2}(?::\d{2})?")
Y_TOL_RATIO = 0.6          # nguong gom dong = 0.6 x chieu cao trung binh


def group_items(anns: list[dict]) -> list[dict]:
    its = [a for a in anns if str(a.get("label", "")).startswith("ITEM_")
           and a.get("box")]
    if not its:
        return []
    heights = [a["box"][3] - a["box"][1] for a in its]
    tol = max(1.0, (sum(heights) / len(heights)) * Y_TOL_RATIO)
    its.sort(key=lambda a: (a["box"][1] + a["box"][3]) / 2.0)

    rows, cur, cy = [], [], None
    for a in its:
        y = (a["box"][1] + a["box"][3]) / 2.0
        if cy is None or abs(y - cy) <= tol:
            cur.append(a)
            cy = y if cy is None else (cy + y) / 2.0
        else:
            rows.append(cur)
            cur, cy = [a], y
    if cur:
        rows.append(cur)

    out = []
    for grp in rows:
        m = {}
        for a in grp:                       # neu trung nhan trong 1 dong, lay cai trai nhat
            lab = a["label"]
            if lab not in m or a["box"][0] < m[lab]["box"][0]:
                m[lab] = a
        name = m.get("ITEM_NAME", {}).get("text")
        if not name:                        # dong khong co ten mon -> bo
            continue
        out.append({"name": name,
                    "unit": None,
                    "quantity": m.get("ITEM_QTY", {}).get("text"),
                    "unit_price": m.get("ITEM_PRICE", {}).get("text"),
                    "amount": m.get("ITEM_AMOUNT", {}).get("text")})
    return out


def label_to_doc(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    anns = d.get("annotations") or []

    def one(label):
        vals = [a.get("text") for a in anns
                if a.get("label") == label and a.get("text")]
        return " ".join(vals).strip() or None

    ts = one("TIMESTAMP") or ""
    md, mt = RE_DATE.search(ts), RE_TIME.search(ts)
    img_id = d.get("file_name") or (p.stem + ".png")

    return normalize_doc({
        "img_id": img_id,
        "quality": None,                    # D2 khong co diem chat luong anh
        "header": {"merchant": one("SELLER"), "address": one("ADDRESS"),
                   "date": md.group(0).replace(" ", "") if md else None,
                   "time": mt.group(0) if mt else None},
        "line_items": group_items(anns),    # CO nhan -> list (co the rong)
        "payment": {"total": one("TOTAL_COST")},
    }, img_id=img_id)


def load_split(split: str) -> tuple[dict[str, dict], dict]:
    """split in {train, val}. Tra ve ({stem: doc}, bao_cao)."""
    d = cfg.D2 / split / "labels"
    files = sorted(d.glob("*.json"))
    docs, n_items = {}, []
    for p in files:
        doc = label_to_doc(p)
        docs[p.stem] = doc
        n_items.append(len(doc["line_items"] or []))
    return docs, {"files": len(files),
                  "items_total": sum(n_items),
                  "items_mean": (sum(n_items) / len(n_items)) if n_items else 0,
                  "items_max": max(n_items) if n_items else 0,
                  "zero_item_docs": sum(1 for x in n_items if x == 0)}
