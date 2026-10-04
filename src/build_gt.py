# -*- coding: utf-8 -*-
"""STAGE 1 - tao split + ground truth cho ca hai bo du lieu.

    python build_gt.py

Khong can GPU. Chay mot lan, ra cac file duoi work/.
"""
import json
import sys

import cfg
from data import splits
from data import gt_mcocr, gt_v3
from schema import validate_doc


def main() -> int:
    print("=" * 74)
    print("1. SPLIT")
    print("=" * 74)
    counts = splits.write_split_files()
    for k in sorted(counts):
        print(f"   {k:12s} {counts[k]:6d} anh   -> work/splits/{k}.txt")
    assert counts["d1_train"] + counts["d1_val"] == counts["d1_all"], "split D1 lech"
    print("   OK: D1 train va val khong giao nhau (da assert trong splits.py)")

    print()
    print("=" * 74)
    print("2. GROUND TRUTH  D1  (MC-OCR)  -  4 rule")
    print("=" * 74)
    d1_docs, rep = gt_mcocr.load_all()
    print(f"   doc {rep['rows_read']} dong CSV  ->  {rep['docs_ok']} doc hop le")
    if rep["dropped"]:
        print(f"   loai {len(rep['dropped'])} dong (anno_num > 12 hoac nhan lech): "
              f"{rep['dropped'][:4]}")
    print("   so truong NULL (dung de do Hallucination rate):")
    for k, v in sorted(rep["null_per_field"].items(), key=lambda x: -x[1])[:8]:
        print(f"      {k:12s} {v:5d}")

    d1_spl = splits.d1_splits()
    for sp in ("train", "val", "all"):
        sub = {k: d1_docs[k] for k in d1_spl[sp] if k in d1_docs}
        out = cfg.GT_DIR / f"d1_{sp}.json"
        out.write_text(json.dumps(sub, ensure_ascii=False), encoding="utf-8")
        print(f"   d1_{sp:5s} {len(sub):6d} doc  -> work/gt/d1_{sp}.json")

    print()
    print("=" * 74)
    print("3. GROUND TRUTH  D2  (vietnamese-receipts-v3)")
    print("=" * 74)
    d2_all = {}
    for sp in ("train", "val"):
        docs, rep2 = gt_v3.load_split(sp)
        d2_all.update(docs)
        out = cfg.GT_DIR / f"d2_{sp}.json"
        out.write_text(json.dumps(docs, ensure_ascii=False), encoding="utf-8")
        print(f"   d2_{sp:5s} {rep2['files']:6d} doc  "
              f"mon/anh: mean={rep2['items_mean']:.1f} max={rep2['items_max']} "
              f"(0 mon: {rep2['zero_item_docs']} anh)  -> work/gt/d2_{sp}.json")
    (cfg.GT_DIR / "d2_all.json").write_text(
        json.dumps(d2_all, ensure_ascii=False), encoding="utf-8")
    print(f"   d2_all   {len(d2_all):6d} doc  -> work/gt/d2_all.json")

    print()
    print("=" * 74)
    print("4. KIEM TRA SCHEMA")
    print("=" * 74)
    bad = 0
    for name, docs in (("d1", d1_docs), ("d2", d2_all)):
        for stem, d in docs.items():
            errs = validate_doc(d)
            if errs:
                bad += 1
                if bad <= 5:
                    print(f"   LOI {name}/{stem}: {errs}")
    print("   doc khong hop le:", bad)

    print()
    print("=" * 74)
    print("5. VAI DOC MAU")
    print("=" * 74)
    for name, docs in (("D1", d1_docs), ("D2", d2_all)):
        k = next(iter(docs))
        d = docs[k]
        print(f"   --- {name}  {k}")
        print(f"       merchant : {d['header']['merchant']!r}")
        print(f"       address  : {str(d['header']['address'])[:52]!r}")
        print(f"       date/time: {d['header']['date']!r} {d['header']['time']!r}")
        print(f"       total    : {d['payment']['total']!r}")
        it = d["line_items"]
        print(f"       items    : {'None (bo data khong co nhan)' if it is None else f'{len(it)} mon'}")
        if it:
            print(f"                  {it[0]}")
    print()
    print("XONG. Buoc tiep: python test_all.py")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
