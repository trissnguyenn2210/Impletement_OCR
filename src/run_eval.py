# -*- coding: utf-8 -*-
"""STAGE 3 - chay MOT phuong phap tren MOT tap test, ghi ra MOT file JSON.

    python run_eval.py --method dummy_perfect --test-on d1_val
    python run_eval.py --method A --train-on d2 --test-on d1_all --limit 50

Moi lan chay = mot dong trong bang. Chay lai mot dong KHONG anh huong dong khac.
"""
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import cfg
from data import load_gt, image_path
from metrics import score_doc, aggregate, item_scores, aggregate_items
from schema import normalize_doc

import methods.dummy            # noqa: F401  - dang ky cac method dummy
try:
    import methods.real         # noqa: F401  - 4 method that
except Exception as _e:
    # KHONG im lang: neu real.py loi cu phap thi phai thay ngay, khong
    # de nguoi dung doan mo vi sao 'chua dang ky method A'.
    print("CANH BAO: khong nap duoc methods/real.py -> chi con cac method dummy.",
          file=sys.stderr)
    print(f"   {type(_e).__name__}: {_e}", file=sys.stderr)
from methods.base import get_method, available


def result_path(method: str, train_on, test_on: str) -> Path:
    return cfg.RESULT_DIR / f"{method}__{train_on or 'none'}__{test_on}.json"


def stratify_by_quality(per_image: list[dict]) -> dict | None:
    """Chia theo anno_image_quality. Chi D1 co truong nay."""
    if not any(d.get("quality") is not None for d in per_image):
        return None
    out = {}
    for name, lo, hi in cfg.QUALITY_BINS:
        sub = [d for d in per_image
               if d.get("quality") is not None and lo <= d["quality"] < hi]
        out[name] = aggregate(sub) if sub else None
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", required=True,
                    help=f"mot trong: {available()}")
    ap.add_argument("--train-on", default=None, choices=[None, "d1", "d2"],
                    help="model duoc train tren bo nao (None = khong train)")
    ap.add_argument("--test-on", required=True, choices=sorted(cfg.TEST_SETS))
    ap.add_argument("--limit", type=int, default=0, help="chi chay N anh dau (de thu)")
    ap.add_argument("--save-pred", action="store_true",
                    help="luu output tho tung anh vao work/pred/ (de debug)")
    a = ap.parse_args()

    gt = load_gt(a.test_on)
    stems = sorted(gt)
    if a.limit and a.limit < len(stems):
        # lay mau NGAU NHIEN (seed co dinh) chu KHONG lay N stem dau.
        # Lay N dau se lech ve mot template: d2 sap xep alphabet nen 300 dau
        # toan la cafe_* -> cac template nay khong in don gia -> price bi lech.
        import random as _rd
        stems = sorted(_rd.Random(42).sample(stems, a.limit))
    print(f"[{a.method}] train_on={a.train_on} test_on={a.test_on} "
          f"n={len(stems)}", flush=True)

    cls = get_method(a.method)
    try:
        model = cls(train_on=a.train_on, test_on=a.test_on)
    except TypeError:
        model = cls(train_on=a.train_on)

    pred_dir = cfg.PRED_DIR / f"{a.method}__{a.train_on or 'none'}__{a.test_on}"
    if a.save_pred:
        pred_dir.mkdir(parents=True, exist_ok=True)

    per_image, per_items, n_fail, t0 = [], [], 0, time.time()
    for i, stem in enumerate(stems, 1):
        g = gt[stem]
        img = image_path(a.test_on, stem)
        try:
            raw = model.predict(img)
        except Exception:
            traceback.print_exc()
            raw = None
        if raw is None:
            n_fail += 1
        pred = normalize_doc(raw, img_id=stem)
        if a.save_pred:
            (pred_dir / f"{stem}.json").write_text(
                json.dumps(pred, ensure_ascii=False, indent=1), encoding="utf-8")
        per_image.append(score_doc(pred, g))
        per_items.append(item_scores(pred.get("line_items"), g.get("line_items")))
        if i % 200 == 0:
            print(f"   {i}/{len(stems)}", flush=True)
    elapsed = time.time() - t0
    model.close()

    agg = aggregate(per_image)
    res = {
        "method": a.method, "train_on": a.train_on, "test_on": a.test_on,
        "n_images": len(stems),
        "parse_fail": 100.0 * n_fail / len(stems) if stems else None,
        "seconds_total": round(elapsed, 2),
        "fps": round(len(stems) / elapsed, 2) if elapsed > 0 else None,
        **model.subtask(), **model.cost(),
        "agg": agg,
        "items": aggregate_items(per_items),
        "by_quality": stratify_by_quality(per_image),
        "per_image": per_image,          # can cho bang quality va de debug
    }
    out = result_path(a.method, a.train_on, a.test_on)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print()
    print(f"   MFS          = {agg['MFS']:.2f} %" if agg["MFS"] is not None else "   MFS = N/A")
    print(f"   Halluc       = {agg['halluc']}" )
    print(f"   parse_fail   = {res['parse_fail']:.2f} %")
    print(f"   FPS          = {res['fps']}")
    print(f"   items        = {'N/A (bo data khong co nhan)' if res['items'] is None else 'co'}")
    print(f"   per_field    = { {k: round(v,1) for k,v in agg['per_field'].items()} }")
    print(f"   n_per_field  = {agg['n_per_field']}")
    print(f"   -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
