# -*- coding: utf-8 -*-
"""STAGE 2 - kiem tra NHANH mot phuong phap tren vai anh truoc khi chay ca bang.

    python self_check.py                        # kiem tra moi truong, khong nap model
    python self_check.py --method A --n 3        # thu method A tren 3 anh
    python self_check.py --method B --n 2 --test-on d2_val

VI SAO CAN FILE NAY:
    Mot dong trong bang MAIN la 1.153 anh. Neu method co loi o anh dau tien,
    chay ca dong roi moi phat hien la mat hang gio GPU. File nay nap model,
    chay N anh, IN RA output tho + diem tung truong de ban nhin bang mat
    xem co hop ly khong.

Thoat 0 = on. Thoat khac 0 = co van de, DUNG chay run_eval.
"""
import argparse
import json
import sys
import traceback

import cfg
from data import image_path, load_gt
from metrics import aggregate, item_scores, score_doc
from schema import normalize_doc, validate_doc


def check_env() -> int:
    """Kiem tra data + GT + thu vien. Khong nap model nang."""
    bad = 0
    print("=" * 72)
    print("1. DUONG DAN")
    print("=" * 72)
    for name, p, must in (("ROOT", cfg.ROOT, True), ("DATA", cfg.DATA, True),
                          ("WORK", cfg.WORK, True),
                          ("D1 csv", cfg.D1_CSV, True),
                          ("D1 images", cfg.D1_IMAGES, True),
                          ("D2 meta", cfg.D2_META, True)):
        ok = p.exists()
        bad += (must and not ok)
        print(f"   {'OK  ' if ok else 'MAT '} {name:12s} {p}")

    print()
    print("=" * 72)
    print("2. GROUND TRUTH (phai chay build_gt.py truoc)")
    print("=" * 72)
    for ts in sorted(cfg.TEST_SETS):
        f = cfg.GT_DIR / f"{ts}.json"
        if not f.exists():
            print(f"   MAT  {ts:9s} -> chay: python build_gt.py")
            bad += 1
            continue
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"   LOI  {ts:9s} {e}")
            bad += 1
            continue
        errs = sum(1 for v in d.values() if validate_doc(v))
        has_items = sum(1 for v in d.values() if v.get("line_items") is not None)
        print(f"   OK   {ts:9s} {len(d):5d} doc   schema loi={errs}   "
              f"co nhan item={has_items}")
        bad += errs > 0

    print()
    print("=" * 72)
    print("3. THU VIEN")
    print("=" * 72)
    import importlib

    for mod, need in (("openpyxl", "fill_excel.py"), ("rapidfuzz", "nhanh hon (tuy chon)"),
                      ("pandas", "doc CSV MC-OCR"), ("PIL", "crop anh"),
                      ("cv2", "method A / D"), ("paddleocr", "method A / C / D"),
                      ("torch", "method B / C / D"), ("transformers", "method B / C / D")):
        try:
            m = importlib.import_module(mod)
            print(f"   OK   {mod:14s} {getattr(m, '__version__', '?'):12s} ({need})")
        except Exception:
            print(f"   --   {mod:14s} {'chua co':12s} ({need})")
            if mod in ("pandas", "PIL"):
                bad += 1
    return bad


def check_method(name: str, test_on: str, n: int) -> int:
    print()
    print("=" * 72)
    print(f"4. THU METHOD {name}  tren {n} anh cua {test_on}")
    print("=" * 72)

    import methods.dummy        # noqa: F401
    try:
        import methods.real     # noqa: F401
    except Exception as e:
        print(f"   methods/real.py KHONG NAP DUOC: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    from methods.base import available, get_method

    if name not in available():
        print(f"   khong co method {name!r}. Da dang ky: {available()}")
        return 1

    gt = load_gt(test_on)
    stems = sorted(gt)[:n]
    print(f"   nap model...")
    try:
        cls = get_method(name)
        try:
            model = cls(train_on=None, test_on=test_on)
        except TypeError:
            model = cls(train_on=None)
    except Exception as e:
        print(f"   NAP MODEL THAT BAI: {type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    print(f"   subtask() = {model.subtask()}")
    print(f"   cost()    = {model.cost()}")

    scores, fails = [], 0
    for stem in stems:
        img = image_path(test_on, stem)
        print()
        print(f"   --- {stem}   {img}")
        if not img or not img.exists():
            print("       ANH KHONG TON TAI")
            fails += 1
            continue
        try:
            raw = model.predict(img)
        except Exception as e:
            print(f"       PREDICT NEM LOI: {type(e).__name__}: {e}")
            traceback.print_exc()
            fails += 1
            continue
        if raw is None:
            print("       predict -> None (parse_fail). Chap nhan duoc voi B/C/D,"
                  " KHONG chap nhan voi A.")
            fails += (name == "A")
        pred = normalize_doc(raw, img_id=stem)
        g = gt[stem]
        s = score_doc(pred, g)
        scores.append(s)
        print(f"       pred  merchant={pred['header']['merchant']!r}")
        print(f"             address ={str(pred['header']['address'])[:46]!r}")
        print(f"             date    ={pred['header']['date']!r}  "
              f"time={pred['header']['time']!r}")
        print(f"             total   ={pred['payment']['total']!r}")
        it = pred["line_items"]
        print(f"             items   ={'None' if it is None else f'{len(it)} mon'}")
        print(f"       gt    merchant={g['header']['merchant']!r}")
        print(f"             date    ={g['header']['date']!r}  total={g['payment']['total']!r}")
        print(f"       diem  {dict((k, round(v, 2)) for k, v in s['per_field'].items())}")
        isc = item_scores(pred.get("line_items"), g.get("line_items"))
        if isc is None:
            print("       item  N/A (gt khong co nhan ITEM_*)")
        else:
            print(f"       item  F1={isc['item_F1']:.2f}  "
                  f"gt={isc['n_gt']} pred={isc['n_pred']} khop={isc['n_matched']}")
    model.close()

    if scores:
        a = aggregate(scores)
        print()
        print(f"   MFS tren {len(scores)} anh = {a['MFS']:.2f} %")
        print(f"   per_field = {dict((k, round(v, 1)) for k, v in a['per_field'].items())}")
    print()
    if fails:
        print(f"   CO {fails} anh LOI -> SUA TRUOC KHI CHAY run_eval.py")
    else:
        print("   KHONG anh nao loi. Co the chay run_eval.py.")
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default=None)
    ap.add_argument("--test-on", default="d1_val", choices=sorted(cfg.TEST_SETS))
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()

    bad = check_env()
    if a.method:
        bad += check_method(a.method, a.test_on, a.n)
    print()
    print("=" * 72)
    print("KET LUAN: " + ("CO VAN DE, xem cac dong MAT / LOI o tren"
                          if bad else "ON"))
    print("=" * 72)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
