# -*- coding: utf-8 -*-
"""Duong dan va hang so dung chung. Ghi de bang bien moi truong khi chay tren server:

    export VIE_DATA=/data/datasets
    export VIE_WORK=/data/work
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("VIE_ROOT", Path(__file__).resolve().parents[1]))
DATA = Path(os.environ.get("VIE_DATA", ROOT / "datasets"))
WORK = Path(os.environ.get("VIE_WORK", ROOT / "work"))

D1 = DATA / "MC-OCR"                        # anh THAT, 4 nhan
D2 = DATA / "vietnamese-receipts-v3"        # anh TONG HOP, 8 nhan

# ---------------------------------------------------------------- D1
D1_CSV = D1 / "mcocr_train_df.csv"
D1_IMAGES = D1 / "train_images" / "train_images"
D1_REC_TRAIN = D1 / "text_recognition_train_data.txt"   # dinh nghia split 922
D1_REC_VAL = D1 / "text_recognition_val_data.txt"       # dinh nghia split 231

# ---------------------------------------------------------------- D2
D2_META = D2 / "metadata.json"

# ---------------------------------------------------------------- work
GT_DIR = WORK / "gt"
SPLIT_DIR = WORK / "splits"
RESULT_DIR = WORK / "results"
PRED_DIR = WORK / "pred"
LOG_DIR = WORK / "logs"

for _d in (GT_DIR, SPLIT_DIR, RESULT_DIR, PRED_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- schema
HEADER_FIELDS = ("merchant", "address", "phone", "tax_code",
                 "invoice_no", "date", "time", "counter", "cashier")
PAYMENT_FIELDS = ("total", "discount", "vat", "tendered", "change")
ITEM_FIELDS = ("name", "unit", "quantity", "unit_price", "amount")

# ten tap test hop le  ->  (bo data, split)
TEST_SETS = {
    "d1_train": ("d1", "train"),
    "d1_val": ("d1", "val"),
    "d1_all": ("d1", "all"),      # 1.153 anh, dung cho T2b/T5b/T8b/T11b
    "d2_train": ("d2", "train"),
    "d2_val": ("d2", "val"),
}

# nguong chia tang chat luong anh (chi D1 co anno_image_quality)
QUALITY_BINS = (("clear", 0.70, 1.01), ("mid", 0.50, 0.70), ("blur", -0.01, 0.50))
