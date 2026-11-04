# -*- coding: utf-8 -*-
"""Duong dan va hang so dung chung.

Co the ghi de bang bien moi truong khi chay tren server:

    export VIE_DATA=/data/datasets
    export VIE_WORK=/data/work

Hoac truyen truc tiep vao cac script co argparse:

    python build_gt.py --data-dir /data/datasets --work-dir /data/work

Thu tu uu tien: tham so CLI > bien moi truong > gia tri mac dinh.
"""
import argparse
import os
from pathlib import Path

# ``cfg.py`` lives in ``src/``, so the default project root is its parent.
_DEFAULT_ROOT = Path(__file__).resolve().parents[1]
_ENV_ROOT = os.environ.get("VIE_ROOT")
_ENV_DATA = os.environ.get("VIE_DATA")
_ENV_WORK = os.environ.get("VIE_WORK")


def add_path_args(parser):
    """Them cac tuy chon duong dan dung chung vao mot parser."""
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=None,
        help="goc project (mac dinh: VIE_ROOT hoac thu muc chua cfg.py)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="thu muc chua MC-OCR/ va vietnamese-receipts-v3/",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="thu muc ghi work/gt, work/results, ...",
    )
    return parser


def configure_paths(args=None):
    """Resolve lai cac path sau khi argparse doc xong.

    Cac module data truy cap cfg.D1/cfg.WORK tai thoi diem chay ham, nen co
    the doi cau hinh sau khi import cfg ma khong can truyen path qua tung ham.
    """
    cli_root = getattr(args, "root_dir", None) if args is not None else None
    cli_data = getattr(args, "data_dir", None) if args is not None else None
    cli_work = getattr(args, "work_dir", None) if args is not None else None

    root_value = cli_root or _ENV_ROOT or _DEFAULT_ROOT
    root = Path(root_value).expanduser()

    data_value = cli_data or _ENV_DATA
    data = Path(data_value).expanduser() if data_value else root / "datasets"

    work_value = cli_work or _ENV_WORK
    work = Path(work_value).expanduser() if work_value else root / "work"

    global ROOT, DATA, WORK
    global D1, D2, D1_CSV, D1_IMAGES, D1_REC_TRAIN, D1_REC_VAL, D2_META
    global GT_DIR, SPLIT_DIR, RESULT_DIR, PRED_DIR, LOG_DIR

    ROOT = root
    DATA = data
    WORK = work

    D1 = DATA / "MC-OCR"
    D2 = DATA / "vietnamese-receipts-v3"

    D1_CSV = D1 / "mcocr_train_df.csv"
    D1_IMAGES = D1 / "train_images" / "train_images"
    D1_REC_TRAIN = D1 / "text_recognition_train_data.txt"
    D1_REC_VAL = D1 / "text_recognition_val_data.txt"

    D2_META = D2 / "metadata.json"

    GT_DIR = WORK / "gt"
    SPLIT_DIR = WORK / "splits"
    RESULT_DIR = WORK / "results"
    PRED_DIR = WORK / "pred"
    LOG_DIR = WORK / "logs"

    for directory in (GT_DIR, SPLIT_DIR, RESULT_DIR, PRED_DIR, LOG_DIR):
        directory.mkdir(parents=True, exist_ok=True)


configure_paths()

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
