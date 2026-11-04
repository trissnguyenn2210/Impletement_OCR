# -*- coding: utf-8 -*-
"""Unit test. Chay: python test_all.py   (khong can GPU, khong can dataset cho
phan metric; phan GT can dataset nen se tu bo qua neu khong tim thay)."""
import sys
import unicodedata

from metrics import anls, numeric_match, norm_date_em, exact, nfc
from metrics.fields import score_doc, aggregate
from metrics.items import item_scores, aggregate_items
from schema import normalize_doc, validate_doc, empty_doc

PASS, FAIL = [], []


def check(name, cond, extra=""):
    (PASS if cond else FAIL).append(name)
    print(("  OK   " if cond else "  FAIL ") + name + (f"   {extra}" if extra else ""))


# ======================================================== core metrics
print("=" * 72)
print("1. METRIC CO BAN")
print("=" * 72)

check("NFC bat buoc: hai cach ma hoa phai cho ANLS = 1.0",
      anls(unicodedata.normalize("NFC", "Tường An"),
           unicodedata.normalize("NFD", "Tường An")) == 1.0)
check("NFC: byte that su khac nhau",
      unicodedata.normalize("NFC", "Tường An") != unicodedata.normalize("NFD", "Tường An"))

check("ANLS trung khit = 1.0", anls("SIEU THI CO.OPMART", "SIEU THI CO.OPMART") == 1.0)
check("ANLS sai 1 ky tu tren 18 -> ~0.94",
      0.93 < anls("SIEU THI CO.OPMARI", "SIEU THI CO.OPMART") < 0.95,
      f"= {anls('SIEU THI CO.OPMARI','SIEU THI CO.OPMART'):.3f}")
check("ANLS cho cap SAI HOAN TOAN van ra 0.667 (ly do phai dung numeric)",
      abs(anls("128000", "285000") - 2 / 3) < 0.01,
      f"= {anls('128000','285000'):.3f}")
check("ANLS duoi nguong 0.5 bi cat ve 0", anls("abcdefgh", "zzzzzzzz") == 0.0)
check("ANLS 'Co.opmart' vs 'Co.op mart' > 0.8 (EM se cho 0)",
      anls("Co.opmart", "Co.op mart") > 0.8)

check("numeric: '166.000' == '166000'", numeric_match("166000", "166.000") == 1.0)
check("numeric: '160.000' != '166.000'  <- cho dung ANLS o day",
      numeric_match("160.000", "166.000") == 0.0)
check("numeric: ANLS cho cap sai tien lai ra 0.857 (ly do phai dung numeric)",
      0.84 < anls("160.000", "166.000") < 0.87,
      f"ANLS = {anls('160.000','166.000'):.3f}")
check("numeric: bo don vi 'VND'", numeric_match("166.000 VNĐ", "166000") == 1.0)
check("numeric: gt rong -> 0", numeric_match("123", None) == 0.0)

check("norm_date: 12/03/2021 == 2021-03-12",
      norm_date_em("2021-03-12", "12/03/2021") == 1.0)
check("norm_date: ANLS chi cho 0.500 cho cap DUNG (mat mot nua diem)",
      abs(anls("2021-03-12 14:32", "12/03/2021 14:32") - 0.5) < 0.01,
      f"ANLS = {anls('2021-03-12 14:32','12/03/2021 14:32'):.3f}")
check("norm_date: ngay khac -> 0", norm_date_em("2021-03-13", "12/03/2021") == 0.0)
check("exact: khac hoa thuong van bang", exact("H:00073344", "h:00073344") == 1.0)
check("nfc(None) = ''", nfc(None) == "")

# ======================================================== fields
print()
print("=" * 72)
print("2. DIEM TRUONG + HALLUCINATION")
print("=" * 72)

gt = normalize_doc({"header": {"merchant": "MINIMART ANAN",
                               "address": "Chợ Sủi Phú Thị Gia Lâm",
                               "date": "11/08/2020", "time": "08:06"},
                    "payment": {"total": "74,000"},
                    "line_items": None}, img_id="x", quality=0.63)

s = score_doc(gt, gt)
check("pred == gt -> moi truong = 1.0", all(v == 1.0 for v in s["per_field"].values()))
check("pred == gt -> halluc_n = 0 (gt null thi pred cung null)", s["halluc_n"] == 0)
check("halluc_d = so truong gt null", s["halluc_d"] == 9,
      f"= {s['halluc_d']} (9 truong null: phone tax_code invoice_no counter cashier discount vat tendered change)")

bad = normalize_doc({"header": {"merchant": "MINIMART ANAN", "phone": "0909123456"},
                     "payment": {}, "line_items": None}, img_id="x")
s2 = score_doc(bad, gt)
check("bia truong gt=null -> halluc_n > 0", s2["halluc_n"] == 1,
      f"halluc_n = {s2['halluc_n']}")

a = aggregate([s])
check("MFS khi dung het = 100", abs(a["MFS"] - 100.0) < 1e-9)
check("n_per_field co mau so RIENG tung truong",
      set(a["n_per_field"]) == {"merchant", "address", "date", "time", "total"})
check("halluc = 0 khi khong bia", a["halluc"] == 0.0)

# ======================================================== items
print()
print("=" * 72)
print("3. LINE ITEMS")
print("=" * 72)

items = [{"name": "Dầu ăn Tường An", "unit": None, "quantity": "2",
          "unit_price": "18.000", "amount": "36.000"},
         {"name": "Sữa Vinamilk", "unit": None, "quantity": "1",
          "unit_price": None, "amount": "62.000"}]

check("gt None -> tra None (N/A, khong phai 0)", item_scores([], None) is None)

r = item_scores(items, items)
check("pred == gt -> item_F1 = 100", abs(r["item_F1"] - 100.0) < 1e-9)
check("REGRESSION: truong gt=None bi LOAI khoi mau so, khong tinh la sai",
      r["price"] == 100.0 and r["_n_price"] == 1,
      f"price = {r['price']}  n = {r['_n_price']} (1 trong 2 mon co unit_price)")

r0 = item_scores([], [])
check("REGRESSION: ca hai rong -> F1 = 100 (du doan dung la 'khong co mon')",
      r0["item_F1"] == 100.0)

r1 = item_scores([], items)
check("pred rong, gt co 2 mon -> F1 = 0", r1["item_F1"] == 0.0)

r2 = item_scores(items + [{"name": "Bịa thêm", "unit": None, "quantity": None,
                           "unit_price": None, "amount": None}], items)
check("bia them 1 mon -> fabricated = 50%", abs(r2["fabricated"] - 50.0) < 1e-9)
check("bia them -> count_ok = 0", r2["count_ok"] == 0.0)

ag = aggregate_items([r, r0, r1])
check("aggregate_items: bo qua cac doc None", ag["n_docs"] == 3)
check("aggregate_items: mau so tung truong cong don dung", ag["n_price"] == 1,
      f"n_price = {ag['n_price']} (chi doc r co mon ghep duoc va co unit_price)")

# ======================================================== schema
print()
print("=" * 72)
print("4. SCHEMA")
print("=" * 72)
check("empty_doc hop le", validate_doc(empty_doc("x")) == [])
check("normalize_doc(None) van ra doc hop le", validate_doc(normalize_doc(None)) == [])
check("normalize_doc loai truong la",
      "zzz" not in normalize_doc({"header": {"zzz": 1}})["header"])
check("normalize_doc: '' -> None", normalize_doc({"header": {"merchant": ""}})["header"]["merchant"] is None)
check("line_items=[] giu nguyen la []", normalize_doc({"line_items": []})["line_items"] == [])
check("line_items=None giu nguyen la None", normalize_doc({"line_items": None})["line_items"] is None)

# ======================================================== GT (can dataset)
print()
print("=" * 72)
print("5. GROUND TRUTH  (bo qua neu khong co dataset)")
print("=" * 72)
try:
    from data import gt_mcocr
    row = {"img_id": "x.jpg", "anno_num": "5", "anno_image_quality": "0.63",
           "anno_texts": "MINIMART ANAN|||Chợ Sủi|||Ngày : 11/08/2020 08:06|||Tổng tiền:|||74,000",
           "anno_labels": "SELLER|||ADDRESS|||TIMESTAMP|||TOTAL_COST|||TOTAL_COST"}
    d = gt_mcocr.row_to_doc(row)
    check("RULE 4: TOTAL_COST lay vung CO SO, bo chu khoa",
          d["payment"]["total"] == "74,000", f"= {d['payment']['total']!r}")
    check("RULE 3: TIMESTAMP strip tien to 'Ngày :' va tach ngay/gio",
          d["header"]["date"] == "11/08/2020" and d["header"]["time"] == "08:06",
          f"= {d['header']['date']!r} {d['header']['time']!r}")
    check("D1 line_items = None (khong co nhan item)", d["line_items"] is None)
    row2 = dict(row, anno_num="25")
    check("RULE 5: loai anh co anno_num > 12", gt_mcocr.row_to_doc(row2) is None)

    from data import gt_v3
    anns = [{"label": "ITEM_NAME", "text": "A", "box": [10, 100, 90, 120]},
            {"label": "ITEM_QTY", "text": "2", "box": [200, 101, 210, 121]},
            {"label": "ITEM_AMOUNT", "text": "36.000", "box": [300, 102, 380, 122]},
            {"label": "ITEM_NAME", "text": "B", "box": [10, 200, 90, 220]},
            {"label": "ITEM_PRICE", "text": "18.000", "box": [240, 201, 300, 221]},
            {"label": "ITEM_AMOUNT", "text": "18.000", "box": [300, 202, 380, 222]}]
    g = gt_v3.group_items(anns)
    check("D2: gom theo KHOANG Y -> dung 2 mon", len(g) == 2, f"= {len(g)}")
    check("D2: mon 1 khong co unit_price van gom dung",
          g[0]["name"] == "A" and g[0]["quantity"] == "2" and g[0]["unit_price"] is None)
    check("D2: mon 2 lay dung unit_price",
          g[1]["name"] == "B" and g[1]["unit_price"] == "18.000")
except Exception as e:
    print(f"  BO QUA phan GT: {type(e).__name__}: {e}")

# ================================================ post-processing cua PRED
print()
print("=" * 72)
print("6. HAU XU LY PREDICTION  (phai khop DUNG rule cua GT)")
print("=" * 72)
from methods.postproc import (clean_timestamp, group_pred_items,
                              parse_json_loose, pick_total, regions_to_doc)

check("PRED RULE 3: strip tien to 'Ngay ban :' va tach ngay/gio",
      clean_timestamp("Ngày bán : 11/08/2020 08:06") == ("11/08/2020", "08:06"))
check("PRED RULE 4: vung TOTAL_COST co ca chu khoa -> lay SO",
      pick_total(["Tổng tiền:", "74,000"]) == "74,000")

_regs = [{"label": "SELLER", "text": "MINIMART", "box": [10, 10, 200, 30]},
         {"label": "SELLER", "text": "ANAN", "box": [210, 10, 300, 30]},
         {"label": "ADDRESS", "text": "Chợ Sủi", "box": [10, 40, 200, 60]},
         {"label": "ADDRESS", "text": "Gia Lâm", "box": [10, 65, 200, 85]},
         {"label": "TIMESTAMP", "text": "Ngày : 11/08/2020 08:06", "box": [10, 90, 300, 110]},
         {"label": "TOTAL_COST", "text": "Tổng tiền:", "box": [10, 120, 100, 140]},
         {"label": "TOTAL_COST", "text": "74,000", "box": [200, 120, 300, 140]}]
_d = regions_to_doc(_regs)
check("PRED: nhieu vung SELLER -> noi bang space (giong GT)",
      _d["header"]["merchant"] == "MINIMART ANAN", f"= {_d['header']['merchant']!r}")
check("PRED: nhieu vung ADDRESS -> noi bang ', ' (giong GT)",
      _d["header"]["address"] == "Chợ Sủi, Gia Lâm", f"= {_d['header']['address']!r}")
check("PRED: total lay '74,000' chu KHONG phai 'Tổng tiền:'",
      _d["payment"]["total"] == "74,000", f"= {_d['payment']['total']!r}")

_it = group_pred_items([{"label": "ITEM_NAME", "text": "A", "box": [10, 100, 90, 120]},
                        {"label": "ITEM_QTY", "text": "2", "box": [200, 101, 210, 121]},
                        {"label": "ITEM_NAME", "text": "B", "box": [10, 200, 90, 220]},
                        {"label": "ITEM_PRICE", "text": "18.000", "box": [240, 201, 300, 221]}])
check("PRED: gom item theo KHOANG Y -> 2 mon", len(_it) == 2, f"= {len(_it)}")
check("PRED: mon thieu don gia van gom dung",
      _it[0]["name"] == "A" and _it[1]["unit_price"] == "18.000")

check("JSON trong ```json fence", parse_json_loose('```json\n{"a":1}\n```') == {"a": 1})
check("JSON co cau dan nhap phia truoc",
      parse_json_loose('Day la ket qua:\n{"a": 2}') == {"a": 2})
check("KHONG co JSON -> None (parse_fail), KHONG phai {}",
      parse_json_loose("khong co json") is None)

check("methods/real.py import duoc KHONG CAN paddle/GPU",
      __import__("importlib").import_module("methods.real") is not None)
from methods.base import available as _av
check("dang ky du 4 method + 5 bien the VLM cho Bang 1",
      {"A", "B", "C", "D", "B_paddleocr_vl", "B_qwen3vl_2b", "B_qwen3vl_4b",
       "B_qwen3vl_8b", "B_internvl35_2b"} <= set(_av()), f"= {_av()}")

# ======================================================== tong ket
print()
print("=" * 72)
print(f"KET QUA:  {len(PASS)} PASS,  {len(FAIL)} FAIL")
print("=" * 72)
if FAIL:
    for f in FAIL:
        print("  FAIL:", f)
sys.exit(1 if FAIL else 0)
