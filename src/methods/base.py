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

_REGISTRY = {}


def register(name):
    def deco(cls):
        _REGISTRY[name] = cls
        return cls
    return deco


def get_method(name):
    if name not in _REGISTRY:
        raise KeyError(f"chua dang ky method {name!r}. Da co: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def available():
    return sorted(_REGISTRY)


class Method(ABC):
    name = "?"

    def __init__(self, train_on=None, **kw):
        self.train_on = train_on          # 'd1' | 'd2' | None
        self.kw = kw
        # Optional trace consumed by the evaluator's per-image visualization.
        # The canonical document remains the only required method output, so
        # methods are free to leave this empty (Method B intentionally does).
        self.last_debug = {}
        self.last_debug_batch = []

    @abstractmethod
    def predict(self, image_path):
        ...

    def predict_batch(self, image_paths):
        """Predict a batch, with a safe sequential fallback.

        VLM methods can override this to use one model forward/generation call
        for several images.  Keeping the fallback here means the evaluator
        remains compatible with the OCR/LLM methods that only implement
        ``predict``.
        """
        outputs = []
        debug_batch = []
        for image_path in image_paths:
            outputs.append(self.predict(image_path))
            debug_batch.append(getattr(self, "last_debug", {}) or {})
        self.last_debug_batch = debug_batch
        self.last_debug = debug_batch[0] if len(debug_batch) == 1 else {}
        return outputs

    def cost(self):
        return {"params_M": None, "gflops": None, "vram_mb": None}

    def subtask(self):
        return {"layout_mAP50": None, "layout_mAP50_95": None,
                "rec_acc": None, "rec_ned": None}

    def close(self):
        pass
