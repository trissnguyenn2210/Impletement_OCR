# -*- coding: utf-8 -*-
"""Interface cho 4 phuong phap.

Tren server, ban chi can hien thuc 4 class ke thua Method va goi register().
Moi thu con lai (nap GT, do metric, ghi JSON, dien Excel) da xong.

HOP DONG:
    predict(image_path) -> canonical dict (xem schema.py) hoac None
        None = that bai parse (chi xay ra voi B, C, D) -> tinh vao parse_fail

    line_items:
        A, C  ->  luon tra None  (khong sinh duoc item)
        B, D  ->  tra list (co the rong)

    cost() -> {"params_M", "gflops", "vram_mb"}  (None neu khong ap dung)
    subtask() -> {"layout_mAP50", "layout_mAP50_95", "rec_acc", "rec_ned"}
        None cho thanh phan ma phuong phap do KHONG CO:
            A: co ca 4        B: None ca 4
            C: co layout, rec = None
            D: layout = None, co rec
"""
from abc import ABC, abstractmethod
from pathlib import Path

_REGISTRY: dict[str, type] = {}


def register(name: str):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_method(name: str):
    if name not in _REGISTRY:
        raise KeyError(f"chua dang ky method {name!r}. Da co: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available() -> list[str]:
    return sorted(_REGISTRY)


class Method(ABC):
    name = "?"

    def __init__(self, train_on: str | None = None, **kw):
        self.train_on = train_on          # 'd1' | 'd2' | None
        self.kw = kw

    @abstractmethod
    def predict(self, image_path: Path) -> dict | None:
        ...

    def cost(self) -> dict:
        return {"params_M": None, "gflops": None, "vram_mb": None}

    def subtask(self) -> dict:
        return {"layout_mAP50": None, "layout_mAP50_95": None,
                "rec_acc": None, "rec_ned": None}

    def close(self):
        pass
