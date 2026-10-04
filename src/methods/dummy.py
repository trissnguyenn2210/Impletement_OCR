# -*- coding: utf-8 -*-
"""Method gia, KHONG dung model nao. Chi de kiem tra toan bo duong ong chay dung
truoc khi cam model that vao.

    dummy_perfect  : copy nguyen GT          -> moi metric phai = 100
    dummy_empty    : tra rong                -> moi metric phai = 0, halluc = 0
    dummy_noisy    : GT + loi co chu dich    -> metric nam giua
    dummy_fail     : luon tra None           -> parse_fail = 100

Chay: python run_eval.py --method dummy_perfect --test-on d1_val
"""
import random
import re
from pathlib import Path

from data import load_gt
from methods.base import Method, register
from schema import empty_doc


class _GTBacked(Method):
    """Lay GT ra lam 'du doan'. Dung DE TEST duong ong, khong phai model."""

    def __init__(self, train_on=None, test_on=None, **kw):
        super().__init__(train_on, **kw)
        self._gt = load_gt(test_on) if test_on else {}

    def _lookup(self, image_path: Path) -> dict | None:
        return self._gt.get(image_path.stem)


@register("dummy_perfect")
class DummyPerfect(_GTBacked):
    name = "dummy_perfect"

    def predict(self, image_path):
        return self._lookup(image_path)


@register("dummy_empty")
class DummyEmpty(_GTBacked):
    name = "dummy_empty"

    def predict(self, image_path):
        return empty_doc(image_path.stem)


@register("dummy_fail")
class DummyFail(_GTBacked):
    name = "dummy_fail"

    def predict(self, image_path):
        return None                     # mo phong JSON parse that bai


@register("dummy_noisy")
class DummyNoisy(_GTBacked):
    """Them 4 loai loi THAT HAY GAP de kiem tra metric phan ung dung:
        merchant : bo mot dau thanh           -> ANLS phai ~0.9, khong phai 0
        date     : doi sang YYYY-MM-DD        -> norm_date_em phai = 1
        total    : bo dau phan cach nghin     -> numeric_match phai = 1
        address  : cat bot duoi               -> ANLS giam nhung > 0.5
    """
    name = "dummy_noisy"

    def predict(self, image_path):
        gt = self._lookup(image_path)
        if gt is None:
            return None
        import copy
        d = copy.deepcopy(gt)
        h, p = d["header"], d["payment"]
        if h.get("merchant"):
            h["merchant"] = h["merchant"].replace("Ị", "I").replace("ầ", "a")
        if h.get("address") and len(h["address"]) > 12:
            h["address"] = h["address"][:-8]
        if h.get("date"):
            m = re.findall(r"\d+", h["date"])
            if len(m) == 3:
                h["date"] = f"{m[2]}-{m[1]}-{m[0]}"
        if p.get("total"):
            p["total"] = re.sub(r"[.,]", "", p["total"])
        return d
