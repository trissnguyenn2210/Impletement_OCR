# -*- coding: utf-8 -*-
"""Metric cho line_items.

line_items la MANG khong dam bao thu tu -> phai GHEP MON truoc, roi moi tinh diem.

Quy uoc N/A:
    gt['line_items'] is None   ->  tra ve None  ->  bang ghi "N/A"
        (bo data KHONG CO NHAN item, vi du D1 MC-OCR)
    pred['line_items'] is None ->  coi nhu []   ->  diem 0
        (phuong phap KHONG SINH DUOC item, vi du A va C)
    Hai truong hop khac nhau: mot la THIEU NHAN, mot la THIEU NANG LUC.

QUAN TRONG - truong con trong mot mon:
    Neu gt cua truong do la None thi KHONG tinh vao mau so, giong nhu truong header.
    Ly do: D2 co so luong nhan KHONG bang nhau giua cac truong
        ITEM_NAME 18.565 · ITEM_AMOUNT 18.565 · ITEM_QTY 17.245 · ITEM_PRICE 13.159
    Mot so template (nhieu quan cafe) khong in don gia. Neu tinh nhung mon do
    thanh 0 diem thi diem price se bi keo xuong vi LY DO SAI.
    Vi vay moi truong item co MAU SO RIENG, bao cao kem trong n_*.
"""
from .core import anls, numeric_match


def _match(pred_items: list[dict], gt_items: list[dict], thr: float):
    """Greedy: moi mon gt lay mon pred co ANLS(name) cao nhat va >= thr."""
    used, pairs = set(), []
    for g in gt_items:
        best, bi = -1.0, None
        for i, p in enumerate(pred_items):
            if i in used:
                continue
            s = anls(p.get("name"), g.get("name"), thr=0.0)   # thr=0 -> diem tho
            if s > best:
                best, bi = s, i
        if bi is not None and best >= thr:
            used.add(bi)
            pairs.append((pred_items[bi], g))
    return pairs


def item_scores(pred_items, gt_items, thr: float = 0.5) -> dict | None:
    if gt_items is None:
        return None                            # KHONG CO NHAN -> N/A
    pred_items = pred_items if isinstance(pred_items, list) else []
    gt_items = gt_items if isinstance(gt_items, list) else []

    pairs = _match(pred_items, gt_items, thr)
    m = len(pairs)
    if not gt_items and not pred_items:
        # CA HAI rong = DU DOAN DUNG (hoa don khong co dong hang nao).
        # D2 co 67/1.005 anh val thuoc nhom receipt_c45_bb khong co mon nao.
        # Neu de F1 = 0 o day thi dummy_perfect chi ra 93.33 thay vi 100.
        P = R = F1 = 1.0
    else:
        P = m / len(pred_items) if pred_items else 0.0
        R = m / len(gt_items) if gt_items else 0.0
        F1 = (2 * P * R / (P + R)) if (P + R) else 0.0

    def field(key, fn):
        """Tra (tong_diem, so_mau). BO QUA mon ma gt cua truong do la None."""
        s, n = 0.0, 0
        for p, g in pairs:
            if g.get(key) in (None, ""):
                continue
            s += fn(p.get(key), g.get(key))
            n += 1
        return s, n

    out = {"item_P": 100.0 * P, "item_R": 100.0 * R, "item_F1": 100.0 * F1,
           "count_ok": float(len(pred_items) == len(gt_items)),
           "n_gt": len(gt_items), "n_pred": len(pred_items), "n_matched": m}
    extra = max(0, len(pred_items) - len(gt_items))
    out["_extra"] = extra
    out["fabricated"] = 100.0 * extra / max(len(gt_items), 1)

    for key, alias, fn in (("name", "name_anls", anls),
                           ("quantity", "qty", numeric_match),
                           ("unit_price", "price", numeric_match),
                           ("amount", "amount", numeric_match)):
        s, n = field(key, fn)
        out[f"_sum_{alias}"] = s
        out[f"_n_{alias}"] = n
        out[alias] = (100.0 * s / n) if n else None      # None = khong co nhan
    return out


def aggregate_items(per_doc: list[dict | None]) -> dict | None:
    """Cong don. Neu MOI anh deu None (khong co nhan) -> tra None."""
    vals = [d for d in per_doc if d is not None]
    if not vals:
        return None

    out = {}
    for k in ("item_P", "item_R", "item_F1"):
        out[k] = sum(v[k] for v in vals) / len(vals)
    # fabricated: tinh theo TONG (so mon bia / so mon gt), khong trung binh %
    _ex = sum(v["_extra"] for v in vals)
    _gt = sum(v["n_gt"] for v in vals)
    out["fabricated"] = 100.0 * _ex / max(_gt, 1)
    out["n_fabricated"] = _ex
    out["count_ok"] = 100.0 * sum(v["count_ok"] for v in vals) / len(vals)

    # cac truong item: cong don TU SO va MAU SO rieng tung truong
    for alias in ("name_anls", "qty", "price", "amount"):
        s = sum(v[f"_sum_{alias}"] for v in vals)
        n = sum(v[f"_n_{alias}"] for v in vals)
        out[alias] = (100.0 * s / n) if n else None
        out[f"n_{alias}"] = n                            # PHAI in ra bang
    out["n_docs"] = len(vals)
    out["n_gt_total"] = sum(v["n_gt"] for v in vals)
    out["n_pred_total"] = sum(v["n_pred"] for v in vals)
    out["n_matched_total"] = sum(v["n_matched"] for v in vals)
    return out
