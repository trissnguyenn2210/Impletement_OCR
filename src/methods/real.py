# -*- coding: utf-8 -*-
"""4 phuong phap THAT. Chay tren server co GPU.

    A  OCR pipeline    layout detector  -> crop -> text det -> CTC rec
    B  VLM zero-shot   ca trang anh     -> VLM  -> JSON
    C  Hybrid          layout detector  -> crop -> VLM doc tung crop
    D  OCR + LLM       text det + rec ca trang (KHONG layout) -> LLM gan truong

Thiet ke 2x2 giai thich vi sao can ca 4:
                        | gan truong bang LAYOUT | gan truong bang NGON NGU
    doc chu bang CTC    |          A             |          D
    doc chu bang VLM    |          C             |          B
    A<->C : doi bo doc chu, giu nguyen cach gan truong
    A<->D : doi cach gan truong, giu nguyen bo doc chu
    B<->D : B co thay pixel, D chi thay text -> do xem "nhin anh" co gia tri gi

IMPORT AN TOAN: moi thu vien nang (paddle, torch, transformers) chi duoc import
BEN TRONG __init__ cua tung class. Nho vay `import methods.real` khong bao gio
lam crash run_eval.py tren may khong co paddle.

TRUOC KHI CHAY CA BANG, bat buoc chay:
    python self_check.py --method A --n 3
de xac nhan method dung duoc tren 3 anh. Dung chay 1.153 anh roi moi phat hien loi.
"""
import os
from pathlib import Path

from .base import Method, register
from .postproc import parse_json_loose, regions_to_doc

# ---------------------------------------------------------------- cau hinh
# Ghi de bang bien moi truong, khong phai sua code:
#   export VIE_VLM=Qwen/Qwen3-VL-4B-Instruct
#   export VIE_LLM=Qwen/Qwen3-4B-Instruct
#   export VIE_LAYOUT_DIR=/path/den/inference      (layout da fine-tune)
#   export VIE_REC_DIR=/path/den/inference         (rec da fine-tune)
VLM_ID = os.environ.get("VIE_VLM", "Qwen/Qwen3-VL-4B-Instruct")
LLM_ID = os.environ.get("VIE_LLM", "Qwen/Qwen3-4B-Instruct")
DEVICE = os.environ.get("VIE_DEVICE", "cuda:0")
MAX_NEW = int(os.environ.get("VIE_MAX_NEW_TOKENS", "768"))

SCHEMA_PROMPT = """Doc hoa don trong anh va tra ve DUNG mot object JSON, khong giai thich gi them.

{
  "header": {"merchant": null, "address": null, "phone": null, "tax_code": null,
             "invoice_no": null, "date": null, "time": null,
             "counter": null, "cashier": null},
  "line_items": [{"name": null, "unit": null, "quantity": null,
                  "unit_price": null, "amount": null}],
  "payment": {"total": null, "discount": null, "vat": null,
              "tendered": null, "change": null}
}

Quy tac:
- Giu NGUYEN VAN chuoi nhu in tren hoa don, giu dau tieng Viet, giu dau phan cach nghin.
- Truong khong co tren hoa don thi de null. TUYET DOI khong doan, khong bia.
- "date" chi chua ngay, "time" chi chua gio.
- "total" la so tien phai tra cuoi cung, khong phai chu "Tong tien".
- Neu hoa don khong co dong hang nao thi "line_items": [].
"""

CROP_PROMPT = ("Vung anh nay la mot phan cua hoa don. Tra ve DUNG van ban in trong anh, "
               "mot dong, khong them chu nao khac, khong giai thich.")


# ===================================================================== tien ich
def _subtask_from_log(train_on: str | None) -> dict:
    """Doc mAP / acc / NED tu file log do CHINH BAN sinh ra khi train.

    KHONG hardcode con so o day. Neu chua co file thi tra None het va bang
    se hien 'N/A' - tot hon la dien so bia.

    Dinh dang work/logs/subtask.json:
        {"layout_d1": {"mAP50": 0.91, "mAP50_95": 0.70},
         "layout_d2": {...},
         "rec_d1":    {"acc": 71.8, "ned": 0.940},
         "rec_d2":    {...}}
    """
    import json

    import cfg

    out = {"layout_mAP50": None, "layout_mAP50_95": None,
           "rec_acc": None, "rec_ned": None}
    p = cfg.LOG_DIR / "subtask.json"
    if not p.exists() or train_on is None:
        return out
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return out
    lay = d.get(f"layout_{train_on}") or {}
    rec = d.get(f"rec_{train_on}") or {}
    out["layout_mAP50"] = lay.get("mAP50")
    out["layout_mAP50_95"] = lay.get("mAP50_95")
    out["rec_acc"] = rec.get("acc")
    out["rec_ned"] = rec.get("ned")
    return out


def _cost_from_log(key: str) -> dict:
    """Doc Params / GFLOPs tu work/logs/cost.json (ban tu do bang paddle.flops
    hoac thop). Khong co thi tra None."""
    import json

    import cfg

    out = {"params_M": None, "gflops": None, "vram_mb": None}
    p = cfg.LOG_DIR / "cost.json"
    if not p.exists():
        return out
    try:
        d = (json.loads(p.read_text(encoding="utf-8")) or {}).get(key) or {}
    except Exception:
        return out
    out.update({k: d.get(k) for k in out if d.get(k) is not None})
    return out


def _model_dir(env: str, default: Path | None) -> str | None:
    v = os.environ.get(env)
    if v:
        return v
    return str(default) if default and Path(default).exists() else None


# ===================================================================== backends
class _Layout:
    """Layout detector. Tra ve [{'label','box','score'}].

    Dung model da fine-tune neu co (outputs/layout/best_model/inference),
    nguoc lai dung PP-DocLayout_plus-L goc.
    """

    def __init__(self, train_on=None):
        import cfg
        from paddleocr import LayoutDetection

        default = cfg.ROOT / "outputs" / "layout" / "best_model" / "inference"
        mdir = _model_dir("VIE_LAYOUT_DIR", default)
        kw = {"model_name": "PP-DocLayout_plus-L", "device": DEVICE}
        if mdir:
            kw["model_dir"] = mdir
        print(f"   [Layout] model_dir={mdir or '(goc, chua fine-tune)'}")
        self.m = LayoutDetection(**kw)

    def __call__(self, img_path):
        res = self.m.predict(str(img_path), batch_size=1)
        out = []
        for r in res:
            for b in (r.get("boxes") or []):
                c = b.get("coordinate") or b.get("bbox")
                if not c:
                    continue
                out.append({"label": str(b.get("label", "")).upper(),
                            "box": [float(x) for x in c[:4]],
                            "score": float(b.get("score", 0.0))})
        return out


class _OCR:
    """Text detection + CTC recognition. Doc chu trong mot anh hoac mot crop."""

    def __init__(self, train_on=None):
        import cfg
        from paddleocr import TextDetection, TextRecognition

        rec_dir = _model_dir("VIE_REC_DIR",
                             cfg.ROOT / "outputs" / "recognition" / "inference")
        self.det = TextDetection(model_name="PP-OCRv5_server_det", device=DEVICE)
        kw = {"model_name": "PP-OCRv5_server_rec", "device": DEVICE}
        if rec_dir:
            kw["model_dir"] = rec_dir
        print(f"   [Rec] model_dir={rec_dir or '(goc, chua fine-tune)'}")
        self.rec = TextRecognition(**kw)

    def lines(self, image) -> list[dict]:
        """image = duong dan hoac numpy array. Tra [{'text','box'}] theo thu tu y."""
        import numpy as np

        src = str(image) if isinstance(image, (str, Path)) else image
        polys = []
        for r in self.det.predict(src, batch_size=1):
            polys.extend(r.get("dt_polys") or [])
        if not polys:
            return []

        img = self._as_array(image)
        crops, boxes = [], []
        for poly in polys:
            p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            x1, y1 = p.min(axis=0)
            x2, y2 = p.max(axis=0)
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2 = min(img.shape[1], int(x2))
            y2 = min(img.shape[0], int(y2))
            if x2 - x1 < 3 or y2 - y1 < 3:
                continue
            crops.append(img[y1:y2, x1:x2])
            boxes.append([x1, y1, x2, y2])
        if not crops:
            return []

        texts = []
        for r in self.rec.predict(crops, batch_size=16):
            texts.append(str(r.get("rec_text", "")))
        n = min(len(texts), len(boxes))
        out = [{"text": texts[i], "box": boxes[i]} for i in range(n)
               if texts[i].strip()]
        return sorted(out, key=lambda d: (d["box"][1], d["box"][0]))

    @staticmethod
    def _as_array(image):
        import cv2
        import numpy as np

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise RuntimeError(f"khong doc duoc anh {image}")
            return img
        return np.asarray(image)


class _VLM:
    """VLM qua transformers. Dung cho B (ca trang) va C (tung crop)."""

    def __init__(self, model_id=VLM_ID):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"   [VLM] {model_id} on {DEVICE}")
        self.proc = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=DEVICE,
            trust_remote_code=True).eval()
        self.torch = torch

    def ask(self, image, prompt: str, max_new=MAX_NEW) -> str:
        from PIL import Image

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True)
        inputs = self.proc(text=[text], images=[image], return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new,
                                      do_sample=False)
        cut = inputs["input_ids"].shape[1]
        return self.proc.batch_decode(ids[:, cut:],
                                      skip_special_tokens=True)[0].strip()


class _LLM:
    """LLM chi doc TEXT (khong thay anh). Dung cho D."""

    def __init__(self, model_id=LLM_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"   [LLM] {model_id} on {DEVICE}")
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, device_map=DEVICE,
            trust_remote_code=True).eval()
        self.torch = torch

    def ask(self, prompt: str, max_new=MAX_NEW) -> str:
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new,
                                      do_sample=False)
        cut = inputs["input_ids"].shape[1]
        return self.tok.batch_decode(ids[:, cut:],
                                     skip_special_tokens=True)[0].strip()


def _crop(img_path, box, pad=4):
    from PIL import Image

    im = Image.open(img_path).convert("RGB")
    x1, y1, x2, y2 = box
    x1 = max(0, int(x1) - pad)
    y1 = max(0, int(y1) - pad)
    x2 = min(im.width, int(x2) + pad)
    y2 = min(im.height, int(y2) + pad)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    return im.crop((x1, y1, x2, y2))


# ===================================================================== A
@register("A")
class A_Pipeline(Method):
    """layout -> crop -> text det -> CTC rec -> gan truong theo NHAN LAYOUT."""

    name = "A  OCR pipeline"

    def __init__(self, train_on=None, **kw):
        super().__init__(train_on, **kw)
        self.layout = _Layout(train_on)
        self.ocr = _OCR(train_on)

    def predict(self, image_path):
        regions = self.layout(image_path)
        out = []
        for r in regions:
            c = _crop(image_path, r["box"])
            if c is None:
                continue
            txt = " ".join(d["text"] for d in self.ocr.lines(c))
            out.append({"label": r["label"], "text": txt, "box": r["box"]})
        doc = regions_to_doc(out)
        doc["line_items"] = None        # A KHONG sinh duoc line_items
        return doc

    def subtask(self):
        return _subtask_from_log(self.train_on)          # co ca 4

    def cost(self):
        return _cost_from_log("A")


# ===================================================================== B
@register("B")
class B_VLM(Method):
    """Ca trang anh -> VLM -> JSON. Khong train, khong crop."""

    name = "B  VLM zero-shot"

    def __init__(self, train_on=None, **kw):
        super().__init__(train_on, **kw)
        self.vlm = _VLM(kw.get("model_id", VLM_ID))

    def predict(self, image_path):
        raw = self.vlm.ask(image_path, SCHEMA_PROMPT)
        d = parse_json_loose(raw)
        if d is None:
            return None                  # -> tinh vao parse_fail
        if d.get("line_items") is None:
            d["line_items"] = []         # B PHAI tra list, [] = "khong co mon"
        return d

    def subtask(self):
        return {"layout_mAP50": None, "layout_mAP50_95": None,
                "rec_acc": None, "rec_ned": None}        # khong co thanh phan nao

    def cost(self):
        return _cost_from_log("B")


# ============================================= cac bien the VLM cho Bang 1
def _mk_vlm(reg_name: str, model_id: str):
    @register(reg_name)
    class _V(B_VLM):
        name = reg_name

        def __init__(self, train_on=None, **kw):
            kw["model_id"] = os.environ.get(
                "VIE_VLM_" + reg_name.split("_", 1)[1].upper(), model_id)
            super().__init__(train_on, **kw)

        def cost(self):
            return _cost_from_log(reg_name)
    _V.__name__ = "VLM_" + reg_name
    return _V


# Bang 1 (sheet VLM) can dung CHINH XAC nhung ten method nay.
_mk_vlm("B_paddleocr_vl", "PaddlePaddle/PaddleOCR-VL")
_mk_vlm("B_qwen3vl_2b", "Qwen/Qwen3-VL-2B-Instruct")
_mk_vlm("B_qwen3vl_4b", "Qwen/Qwen3-VL-4B-Instruct")
_mk_vlm("B_qwen3vl_8b", "Qwen/Qwen3-VL-8B-Instruct")
_mk_vlm("B_internvl35_2b", "OpenGVLab/InternVL3_5-2B")


# ===================================================================== C
@register("C")
class C_Hybrid(Method):
    """layout -> crop -> VLM doc TUNG CROP (thay cho CTC rec).

    Giong A o cach gan truong, khac A o bo doc chu. So A voi C = do RIENG
    anh huong cua bo doc chu.
    """

    name = "C  Hybrid"

    def __init__(self, train_on=None, **kw):
        super().__init__(train_on, **kw)
        self.layout = _Layout(train_on)
        self.vlm = _VLM(kw.get("model_id", VLM_ID))

    def predict(self, image_path):
        out = []
        for r in self.layout(image_path):
            c = _crop(image_path, r["box"])
            if c is None:
                continue
            txt = self.vlm.ask(c, CROP_PROMPT, max_new=64)
            out.append({"label": r["label"], "text": txt, "box": r["box"]})
        doc = regions_to_doc(out)
        doc["line_items"] = None        # C KHONG sinh duoc line_items
        return doc

    def subtask(self):
        s = _subtask_from_log(self.train_on)
        s["rec_acc"] = s["rec_ned"] = None      # C khong dung CTC rec
        return s

    def cost(self):
        return _cost_from_log("C")


# ===================================================================== D
@register("D")
class D_OCR_LLM(Method):
    """text det + CTC rec CA TRANG (KHONG layout) -> LLM gan truong.

    LLM chi thay TEXT, khong thay pixel. So D voi B = do xem "nhin thay anh"
    dong gop bao nhieu, khi ca hai deu gan truong bang ngon ngu.
    """

    name = "D  OCR + LLM"

    def __init__(self, train_on=None, **kw):
        super().__init__(train_on, **kw)
        self.ocr = _OCR(train_on)
        self.llm = _LLM(kw.get("model_id", LLM_ID))

    def predict(self, image_path):
        lines = self.ocr.lines(image_path)
        if not lines:
            return None
        # giu toa do: thu tu dong va vi tri x la tin hieu DUY NHAT con lai
        # de LLM suy ra cot nao la so luong, cot nao la thanh tien.
        body = "\n".join(f"[x={int(d['box'][0])},y={int(d['box'][1])}] {d['text']}"
                         for d in lines)
        prompt = (SCHEMA_PROMPT
                  + "\nDuoi day la cac dong chu da doc duoc tu hoa don, kem toa do:\n"
                  + body + "\n\nJSON:")
        d = parse_json_loose(self.llm.ask(prompt))
        if d is None:
            return None
        if d.get("line_items") is None:
            d["line_items"] = []
        return d

    def subtask(self):
        s = _subtask_from_log(self.train_on)
        s["layout_mAP50"] = s["layout_mAP50_95"] = None   # D khong dung layout
        return s

    def cost(self):
        return _cost_from_log("D")
