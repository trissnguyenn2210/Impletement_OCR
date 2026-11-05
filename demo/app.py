# -*- coding: utf-8 -*-
"""Web demo chay 4 phuong phap trich xuat thong tin hoa don (A, B, C, D).

    A  OCR pipeline   layout (fine-tune) -> crop -> text det -> CTC rec (fine-tune)
    B  VLM zero-shot  ca trang anh -> VLM -> JSON
    C  Hybrid         layout (fine-tune) -> crop -> VLM doc tung crop
    D  OCR + LLM      text det + CTC rec ca trang -> LLM gan truong

Chay tai thu muc goc project:

    python demo/app.py                       # http://127.0.0.1:7860
    python demo/app.py --share               # tao link public gradio
    python demo/app.py --device cpu          # ep chay CPU

Model logic nam o ``src/methods/real.py`` (ban da sua loi). File nay chi nap
model mot lan, chia se backend giua cac method (A/C dung chung layout, A/D
dung chung OCR) va hien thi ket qua.
"""
import argparse
import gc
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
CKPT = ROOT / "checkpoints"

VLM_CHOICES = [
    "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen/Qwen3-VL-8B-Instruct",
    "OpenGVLab/InternVL3_5-2B",
    "PaddlePaddle/PaddleOCR-VL",
]
LLM_CHOICES = [
    "Qwen/Qwen3-4B-Instruct-2507",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen2.5-1.5B-Instruct",
    "Qwen/Qwen2.5-0.5B-Instruct",
]
METHOD_INFO = {
    "A": ("OCR pipeline", "Layout fine-tune → crop → text det → CTC rec fine-tune → gán trường theo nhãn layout"),
    "B": ("VLM zero-shot", "Cả trang ảnh → VLM → JSON (không train)"),
    "C": ("Layout + VLM", "Layout fine-tune → crop → VLM đọc từng vùng → gán trường theo nhãn layout"),
    "D": ("OCR + LLM", "Text det + CTC rec fine-tune cả trang → LLM (chỉ thấy text) → JSON"),
}
LABEL_COLORS = {
    "SELLER": (220, 50, 47),
    "ADDRESS": (38, 139, 210),
    "TIMESTAMP": (133, 153, 0),
    "TOTAL_COST": (211, 54, 130),
    "TOTAL": (211, 54, 130),
    "PHONE": (42, 161, 152),
    "ITEM_NAME": (181, 137, 0),
    "ITEM_QTY": (108, 113, 196),
    "ITEM_PRICE": (203, 75, 22),
    "ITEM_AMOUNT": (88, 110, 117),
}
OCR_COLOR = (230, 120, 20)
SHOW_FIELDS = (
    ("header", "merchant"), ("header", "address"), ("header", "phone"),
    ("header", "tax_code"), ("header", "invoice_no"), ("header", "date"),
    ("header", "time"), ("header", "counter"), ("header", "cashier"),
    ("payment", "total"), ("payment", "discount"), ("payment", "vat"),
    ("payment", "tendered"), ("payment", "change"),
)


# --------------------------------------------------------------------- env
def configure_runtime(device="auto", max_new_tokens=None):
    """Dat bien moi truong TRUOC khi import methods.real (no doc luc import)."""
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("VIE_ROOT", str(ROOT))
    # Checkpoint fine-tune cua layout + recognition (giai nen tu archive).
    for env, sub in (("VIE_LAYOUT_DIR", "layout"), ("VIE_REC_DIR", "recognition")):
        cand = CKPT / sub / "inference"
        if not os.environ.get(env) and (cand / "inference.pdiparams").is_file():
            os.environ[env] = str(cand)
    if max_new_tokens:
        os.environ["VIE_MAX_NEW_TOKENS"] = str(max_new_tokens)

    paddle_dev, torch_dev, dtype = "cpu", "cpu", "bfloat16"
    # Windows: torch PHAI duoc import truoc paddle, nguoc lai torch\lib\shm.dll
    # loi "WinError 127" do xung dot DLL.
    try:
        import torch
        if device != "cpu" and torch.cuda.is_available():
            torch_dev = "cuda:0"
            dtype = "bfloat16" if torch.cuda.is_bf16_supported() else "float16"
    except Exception:
        pass
    try:
        import paddle  # noqa: F401
        if (device != "cpu" and paddle.device.is_compiled_with_cuda()
                and paddle.device.cuda.device_count() > 0):
            paddle_dev = "gpu:0"
    except Exception:
        pass
    os.environ.setdefault("VIE_PADDLE_DEVICE", paddle_dev)
    os.environ.setdefault("VIE_TORCH_DEVICE", torch_dev)
    os.environ.setdefault("VIE_TORCH_DTYPE", dtype)
    if str(SRC) not in sys.path:
        sys.path.insert(0, str(SRC))


# --------------------------------------------------------------------- models
class Backends:
    """Nap moi backend 1 lan, dung chung giua cac method."""

    def __init__(self):
        self.layout = None
        self.ocr = None
        self.vlm = {}
        self.llm = {}

    @property
    def R(self):
        import methods.real as real
        return real

    def get_layout(self):
        if self.layout is None:
            self.layout = self.R._Layout()
        return self.layout

    def get_ocr(self):
        if self.ocr is None:
            self.ocr = self.R._OCR()
        return self.ocr

    def get_vlm(self, model_id):
        if model_id not in self.vlm:
            # Chi giu 1 VLM trong bo nho.
            self.vlm.clear()
            self._free()
            self.vlm[model_id] = self.R._VLM(model_id)
        return self.vlm[model_id]

    def get_llm(self, model_id):
        if model_id not in self.llm:
            self.llm.clear()
            self._free()
            self.llm[model_id] = _CapturingLLM(self.R._LLM(model_id))
        return self.llm[model_id]

    def unload(self, keep=()):
        if "layout" not in keep:
            self.layout = None
        if "ocr" not in keep:
            self.ocr = None
        if "vlm" not in keep:
            self.vlm.clear()
        if "llm" not in keep:
            self.llm.clear()
        self._free()

    @staticmethod
    def _free():
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            import paddle
            if paddle.device.is_compiled_with_cuda():
                paddle.device.cuda.empty_cache()
        except Exception:
            pass

    def status(self):
        parts = []
        if self.layout is not None:
            parts.append("Layout")
        if self.ocr is not None:
            parts.append("OCR det+rec")
        parts += [f"VLM {k}" for k in self.vlm]
        parts += [f"LLM {k}" for k in self.llm]
        return ", ".join(parts) or "(chưa nạp model nào)"


class _CapturingLLM:
    """Boc _LLM de giu lai output tho (D khong luu raw vao last_debug)."""

    def __init__(self, llm):
        self.llm = llm
        self.last_raw = None

    def ask(self, prompt, max_new=None):
        kw = {} if max_new is None else {"max_new": max_new}
        self.last_raw = self.llm.ask(prompt, **kw)
        return self.last_raw


BACKENDS = Backends()


def build_method(key, vlm_id, llm_id):
    """Tao instance method dung backend chung thay vi nap lai model."""
    R = BACKENDS.R
    cls = {"A": R.A_Pipeline, "B": R.B_VLM, "C": R.C_Hybrid, "D": R.D_OCR_LLM}[key]
    m = cls.__new__(cls)
    R.Method.__init__(m, train_on="d1", model_id=vlm_id if key in "BC" else llm_id)
    if key == "A":
        m.layout = BACKENDS.get_layout()
        m.ocr = BACKENDS.get_ocr()
        m.layout_batch_size = 1
    elif key == "B":
        m.vlm = BACKENDS.get_vlm(vlm_id)
    elif key == "C":
        m.layout = BACKENDS.get_layout()
        m.vlm = BACKENDS.get_vlm(vlm_id)
    elif key == "D":
        m.ocr = BACKENDS.get_ocr()
        m.llm = BACKENDS.get_llm(llm_id)
    return m


# --------------------------------------------------------------------- drawing
def _font(size):
    from PIL import ImageFont

    for path in (
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ):
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def draw_trace(image_path, debug):
    from PIL import Image, ImageDraw

    im = Image.open(image_path).convert("RGB")
    debug = debug or {}
    regions = debug.get("regions") or []
    lines = debug.get("ocr_lines") or []
    if not regions and not lines:
        return im
    draw = ImageDraw.Draw(im, "RGBA")
    width = max(2, round(max(im.size) / 500))
    font = _font(max(12, round(max(im.size) / 70)))

    def box_label(box, color, caption):
        x1, y1, x2, y2 = [float(v) for v in box[:4]]
        draw.rectangle((x1, y1, x2, y2), outline=color + (255,), fill=color + (40,), width=width)
        tb = draw.textbbox((0, 0), caption, font=font)
        th = tb[3] - tb[1] + 6
        top = y1 - th if y1 - th >= 0 else y2
        draw.rectangle((x1, top, x1 + tb[2] + 8, top + th), fill=color + (230,))
        draw.text((x1 + 4, top + 2), caption, fill="white", font=font)

    for i, r in enumerate(regions, 1):
        lab = str(r.get("label") or "REGION").upper()
        box_label(r["box"], LABEL_COLORS.get(lab, (90, 90, 200)), f"{i}. {lab}")
    if not regions:
        for i, ln in enumerate(lines, 1):
            box_label(ln["box"], OCR_COLOR, str(i))
    return im


# --------------------------------------------------------------------- run
def prepare_image(path, max_side):
    """Xoay theo EXIF, chuyen RGB va luu ra ten ASCII (cv2.imread tren Windows
    khong doc duoc duong dan unicode)."""
    from PIL import Image, ImageOps

    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.Resampling.LANCZOS)
    out_dir = Path(tempfile.gettempdir()) / "vie_demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"input_{int(time.time() * 1000)}.jpg"
    im.save(out, quality=95)
    return out


def trace_text(key, debug, method):
    debug = debug or {}
    out = []
    if debug.get("regions"):
        out.append("# Layout regions (label | text)")
        for i, r in enumerate(debug["regions"], 1):
            out.append(f"{i:>2}. {r.get('label')}: {r.get('text')}")
    if debug.get("ocr_lines"):
        out.append("# OCR lines (det + rec)")
        for i, ln in enumerate(debug["ocr_lines"], 1):
            x1, y1 = int(ln["box"][0]), int(ln["box"][1])
            out.append(f"{i:>3}. [x={x1},y={y1}] {ln.get('text')}")
    if debug.get("raw_text"):
        out.append("# VLM raw output")
        out.append(str(debug["raw_text"]))
    if key == "D":
        raw = getattr(getattr(method, "llm", None), "last_raw", None)
        if raw:
            out.append("\n# LLM raw output")
            out.append(raw)
    return "\n".join(out) or "(không có trace)"


def run_one(key, img_path, vlm_id, llm_id):
    from schema import normalize_doc

    t0 = time.perf_counter()
    m = build_method(key, vlm_id, llm_id)
    t_load = time.perf_counter() - t0
    t1 = time.perf_counter()
    raw_doc = m.predict(img_path)
    t_pred = time.perf_counter() - t1
    debug = getattr(m, "last_debug", {}) or {}
    doc = normalize_doc(raw_doc, img_id=Path(img_path).stem) if raw_doc is not None else None
    if doc is not None and key in "AC":
        doc["line_items"] = None
    return {
        "doc": doc,
        "parse_fail": raw_doc is None,
        "image": draw_trace(img_path, debug),
        "trace": trace_text(key, debug, m),
        "t_load": t_load,
        "t_pred": t_pred,
    }


# Truong co nhan trong dataset MC-OCR (4 nhan layout) -> mau giong khung tren anh.
# TIMESTAMP duoc tach thanh 2 truong date + time.
DATASET_FIELDS = {
    "merchant": "SELLER",
    "address": "ADDRESS",
    "date": "TIMESTAMP",
    "time": "TIMESTAMP",
    "total": "TOTAL_COST",
}


def _esc(value):
    import html

    return html.escape(str(value)).replace("\n", "<br>")


def _field_cell(field):
    lab = DATASET_FIELDS.get(field)
    if lab:
        r, g, b = LABEL_COLORS[lab]
        return (f'<td class="fname ds" style="color:rgb({r},{g},{b})" '
                f'title="Có trong dataset (nhãn {lab})">{field}</td>')
    return f'<td class="fname" title="Không có trong dataset">{field}</td>'


def fields_html(doc):
    rows = []
    for blk, f in SHOW_FIELDS:
        value = ((doc or {}).get(blk) or {}).get(f) or ""
        cls = ' class="ds"' if f in DATASET_FIELDS else ""
        rows.append(f"<tr>{_field_cell(f)}<td{cls}>{_esc(value)}</td></tr>")
    return ('<table class="vie-table"><thead><tr><th>Trường</th><th>Giá trị</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table>')


def items_html(doc):
    items = (doc or {}).get("line_items")
    if items is None:
        return '<div class="vie-note">Line items: phương pháp này không trích xuất line items.</div>'
    if not items:
        return '<div class="vie-note">Line items: không có món nào.</div>'
    cols = ("name", "unit", "quantity", "unit_price", "amount")
    head = "".join(f"<th>{c}</th>" for c in ("#",) + cols)
    body = "".join(
        "<tr><td>{}</td>{}</tr>".format(i + 1, "".join(f"<td>{_esc(it.get(c) or '')}</td>" for c in cols))
        for i, it in enumerate(items))
    return (f'<div class="vie-sub">Line items ({len(items)})</div>'
            f'<table class="vie-table"><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>')


def summary_html(results):
    keys = [k for k in "ABCD" if k in results]
    if not keys:
        return '<div class="vie-note">Chưa có kết quả. Tải ảnh lên, chọn phương pháp rồi bấm Chạy.</div>'
    head = "<th>Trường</th>" + "".join(f"<th>{k} {METHOD_INFO[k][0]}</th>" for k in keys)
    rows = []
    for blk, f in SHOW_FIELDS:
        cells = []
        for k in keys:
            res = results[k]
            if res.get("error"):
                cells.append('<td class="bad">⚠ lỗi</td>')
            elif res.get("doc") is None:
                cells.append('<td class="bad">✗ parse fail</td>')
            else:
                cls = ' class="ds"' if f in DATASET_FIELDS else ""
                cells.append(f"<td{cls}>{_esc((res['doc'].get(blk) or {}).get(f) or '')}</td>")
        rows.append(f"<tr>{_field_cell(f)}{''.join(cells)}</tr>")
    n_items, times = [], []
    for k in keys:
        res = results[k]
        items = (res.get("doc") or {}).get("line_items")
        n_items.append(f"<td>{'N/A' if items is None else len(items)}</td>")
        times.append(f"<td>{'' if res.get('error') else format(res['t_pred'], '.1f')}</td>")
    rows.append(f'<tr class="meta"><td class="fname">line_items (số món)</td>{"".join(n_items)}</tr>')
    rows.append(f'<tr class="meta"><td class="fname">thời gian dự đoán (s)</td>{"".join(times)}</tr>')
    return (f'<table class="vie-table"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
            + LEGEND_HTML)


def _legend():
    pills = "".join(
        f'<span class="lg-pill" style="border-color:rgb{LABEL_COLORS[lab]}">'
        f'<span class="lg-sw" style="background:rgb{LABEL_COLORS[lab]}"></span>'
        f'<b style="color:rgb{LABEL_COLORS[lab]}">{f}</b><span class="lg-lab">{lab}</span></span>'
        for f, lab in DATASET_FIELDS.items())
    return ('<div class="vie-legend">'
            '<div class="lg-title">Chú thích</div>'
            f'<div class="lg-pills">{pills}</div>'
            '<div class="lg-row"><b>Chữ đậm, có màu:</b> trường có nhãn trong tập MC-OCR, được dùng để '
            'đánh giá. Màu trùng với màu khung vùng trên ảnh; TIMESTAMP được tách thành date và time.</div>'
            '<div class="lg-row"><span class="lg-plain">Chữ thường:</span> trường nằm ngoài tập nhãn của '
            'dataset, chỉ mang tính tham khảo.</div>'
            '</div>')


LEGEND_HTML = _legend()


# --------------------------------------------------------------------- pipeline figures
# kind: io | paddle (model Paddle) | vlm (VLM/LLM) | rule (xu ly quy tac)
PIPELINES = {
    "A": [("io", "Ảnh hoá đơn", ""),
          ("paddle", "Layout detection", "PP-DocLayout+ · fine-tune"),
          ("rule", "Crop vùng", "SELLER / ADDRESS / TIMESTAMP / TOTAL_COST"),
          ("paddle", "Text detection", "PP-OCRv5 det"),
          ("paddle", "Text recognition", "CTC · PP-OCRv5 rec · fine-tune"),
          ("rule", "Gán trường", "theo nhãn layout"),
          ("io", "JSON", "không có line items")],
    "B": [("io", "Ảnh hoá đơn", "cả trang"),
          ("vlm", "VLM", "Qwen3-VL / InternVL · zero-shot"),
          ("rule", "Parse JSON", "prompt theo schema"),
          ("io", "JSON", "có line items")],
    "C": [("io", "Ảnh hoá đơn", ""),
          ("paddle", "Layout detection", "PP-DocLayout+ · fine-tune"),
          ("rule", "Crop vùng", "SELLER / ADDRESS / TIMESTAMP / TOTAL_COST"),
          ("vlm", "VLM đọc từng vùng", "thay cho det + CTC rec"),
          ("rule", "Gán trường", "theo nhãn layout"),
          ("io", "JSON", "không có line items")],
    "D": [("io", "Ảnh hoá đơn", "cả trang, không layout"),
          ("paddle", "Text detection", "PP-OCRv5 det"),
          ("paddle", "Text recognition", "CTC · PP-OCRv5 rec · fine-tune"),
          ("rule", "Ghép dòng + toạ độ", "[x, y] text"),
          ("vlm", "LLM", "Qwen3 · chỉ thấy text"),
          ("rule", "Parse JSON", ""),
          ("io", "JSON", "có line items")],
}
KIND_NAMES = {"paddle": "Model Paddle", "vlm": "VLM / LLM", "rule": "Xử lý quy tắc", "io": "Vào / ra"}


def pipeline_html(key):
    blocks = []
    for i, (kind, title, sub) in enumerate(PIPELINES[key]):
        if i:
            blocks.append('<div class="pl-arrow">➜</div>')
        sub_html = f'<div class="pl-s">{sub}</div>' if sub else ""
        blocks.append(f'<div class="pl-b pl-{kind}"><div class="pl-t">{title}</div>{sub_html}</div>')
    legend = "".join(f'<span class="pl-key pl-{k}">{v}</span>' for k, v in KIND_NAMES.items())
    return (f'<div class="pl"><div class="pl-row">{"".join(blocks)}</div>'
            f'<div class="pl-legend">Pipeline {key}: {legend}</div></div>')


CSS = """
html { zoom: 0.8; }
.gradio-container { max-width: 100% !important; padding: 6px 16px !important; }
footer { display: none !important; }
.gradio-container h1 { margin: 0 0 4px 0; }
.vie-table { width: 100%; border-collapse: collapse; font-size: 14px; }
.vie-table th, .vie-table td { border: 1px solid var(--border-color-primary); padding: 4px 8px;
  text-align: left; vertical-align: top; word-break: break-word; }
.vie-table th { background: var(--background-fill-secondary); font-weight: 600; }
.vie-table td.fname { white-space: nowrap; font-weight: 400; color: var(--body-text-color-subdued); }
.vie-table td.fname.ds { font-weight: 700; }
.vie-table td.ds { font-weight: 700; }
.vie-table td.bad { color: #c0392b; }
.vie-table tr.meta td { font-style: italic; }
.vie-note, .vie-legend { font-size: 13px; color: var(--body-text-color-subdued); margin: 6px 0; }
.vie-sub { font-weight: 600; margin: 8px 0 4px; }
.vie-legend { border: 1px solid var(--border-color-primary); border-radius: 8px; padding: 8px 10px;
  background: var(--background-fill-secondary); }
.lg-title { font-weight: 700; color: var(--body-text-color); margin-bottom: 6px; }
.lg-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 6px; }
.lg-pill { display: inline-flex; align-items: center; gap: 5px; border: 1px solid; border-radius: 999px;
  padding: 1px 9px; font-size: 12px; }
.lg-sw { width: 9px; height: 9px; border-radius: 2px; display: inline-block; }
.lg-lab { font-size: 11px; opacity: 0.75; font-family: var(--font-mono); }
.lg-row { margin-top: 2px; line-height: 1.45; }
.lg-row b { color: var(--body-text-color); }
.lg-plain { color: var(--body-text-color); font-weight: 400; }
.vie-scroll { overflow-y: auto; }
#summary-box { max-height: 108vh; }
.fields-box { max-height: 82vh; }
.pl { margin-top: 6px; }
.ex-gallery, .ex-gallery .grid-wrap { height: auto !important; max-height: none !important; overflow: visible !important; }
.ex-gallery .thumbnail-item { height: 130px !important; display: flex !important; flex-direction: column !important;
  overflow: hidden; }
.ex-gallery .thumbnail-item img { flex: 1 1 auto; min-height: 0; width: 100%; object-fit: cover; cursor: pointer; }
.ex-gallery .caption-label { position: static !important; transform: none !important; max-width: 100% !important;
  width: 100%; margin: 0 !important; border-radius: 0 !important; font-size: 11px; line-height: 1.25;
  white-space: normal !important; text-align: center; padding: 3px 2px !important; }
.pl-row { display: flex; align-items: stretch; gap: 6px; flex-wrap: wrap; }
.pl-b { flex: 1 1 0; min-width: 110px; border-radius: 8px; padding: 8px 10px; border: 2px solid;
  display: flex; flex-direction: column; justify-content: center; text-align: center; }
.pl-t { font-weight: 700; font-size: 14px; }
.pl-s { font-size: 12px; opacity: 0.85; margin-top: 2px; }
.pl-arrow { align-self: center; font-size: 18px; color: var(--body-text-color-subdued); }
.pl-paddle { border-color: #268bd2; background: rgba(38,139,210,0.12); }
.pl-vlm { border-color: #9b59b6; background: rgba(155,89,182,0.14); }
.pl-rule { border-color: #7f8c8d; background: rgba(127,140,141,0.12); border-style: dashed; }
.pl-io { border-color: #2e9e6b; background: rgba(46,158,107,0.12); }
.pl-legend { font-size: 12px; color: var(--body-text-color-subdued); margin-top: 6px; }
.pl-key { border: 1px solid; border-radius: 4px; padding: 0 6px; margin-left: 6px; }
"""


# Anh mau chon tu ket qua danh gia d1_val, da kiem tra lai bang chinh demo (Method A,
# cham voi ground truth). Anh goc lay tu kie_data/images. Luu y: demo xoay anh theo EXIF
# nen mot so anh "xoay 90 do" trong file danh gia cu lai cho ket qua dung o day.
_EX_DIR = ROOT / "datasets" / "MC-OCR" / "kie_data" / "kie_data" / "images"
# 4 anh dau: ket qua tot (A tren demo 0.95 - 0.99, B/C/D trong danh gia >= 0.97).
GOOD_EXAMPLES = [
    (_EX_DIR / "mcocr_public_145013tffpu.jpg", "MINIMART ANAN"),
    (_EX_DIR / "mcocr_public_145013bfddt.jpg", "VinCommerce"),
    (_EX_DIR / "mcocr_public_145014yfnqu.jpg", "FamilyMart"),
    (_EX_DIR / "mcocr_public_145013bykuv.jpg", "THUC COFFEE"),
]
# 4 anh sau: ket qua chua tot, moi anh mot van de.
HARD_EXAMPLES = [
    # A = 0.00: giay in nhiet phai mau, chu nhoe -> OCR ra chuoi vo nghia.
    (_EX_DIR / "mcocr_public_145013vbjnr.jpg", "chu in nhiet mo"),
    # A = 0.22: anh chat luong thap -> doc nham/thieu chu so ("/05/2020", "88:23:04").
    (_EX_DIR / "mcocr_public_145014bpeci.jpg", "doc nham chu so"),
    # A = 0.51: layout gan nham nhan, so tien "50.000 D" bi gan TIMESTAMP -> mat total.
    (_EX_DIR / "mcocr_public_145014ofdxo.jpg", "gan nham nhan layout"),
    # A = 0.25: ten cua hang nam o 2 vung, ghep sai thu tu -> "5156 VINMART TEN DAI LY".
    (_EX_DIR / "mcocr_public_145014lijow.jpg", "ghep sai thu tu vung"),
]


# --------------------------------------------------------------------- UI
def build_ui(args):
    import gradio as gr

    # 4 anh dau ket qua tot, 4 anh sau moi anh mot van de (xem GOOD/HARD_EXAMPLES).
    samples = [str(p) for p, _ in GOOD_EXAMPLES + HARD_EXAMPLES if p.is_file()]

    with gr.Blocks(title="Receipt VIE Demo", fill_width=True) as ui:
        gr.Markdown("# 🧾 Receipt VIE: Demo 4 phương pháp")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
                img_in = gr.Image(type="filepath", label="Ảnh hoá đơn", height="42vh")
                galleries = []
                if samples:
                    g = gr.Gallery(samples, label="Ảnh mẫu", show_label=False, columns=4, rows=2,
                                   height=215, allow_preview=False, object_fit="cover",
                                   elem_classes="ex-gallery")
                    galleries.append((g, samples))
                methods_in = gr.CheckboxGroup(
                    [(f"{k} {v[0]}", k) for k, v in METHOD_INFO.items()],
                    value=["A"], label="Phương pháp")
                with gr.Accordion("Cấu hình model", open=False):
                    vlm_in = gr.Dropdown(VLM_CHOICES, value=args.vlm, allow_custom_value=True,
                                         label="VLM (dùng cho B, C)")
                    llm_in = gr.Dropdown(LLM_CHOICES, value=args.llm, allow_custom_value=True,
                                         label="LLM (dùng cho D)")
                    max_side_in = gr.Slider(0, 3000, value=args.max_side, step=100,
                                            label="Resize cạnh dài tối đa (0 = giữ nguyên)")
                    low_mem_in = gr.Checkbox(value=args.low_mem,
                                             label="Tiết kiệm bộ nhớ: giải phóng model sau mỗi phương pháp")
                with gr.Row():
                    run_btn = gr.Button("▶ Chạy", variant="primary", scale=2)
                    unload_btn = gr.Button("Giải phóng model", scale=1)
                status = gr.Markdown(f"Model đang nạp: {BACKENDS.status()}")
            with gr.Column(scale=4):
                with gr.Tab("So sánh"):
                    summary = gr.HTML(summary_html({}), elem_id="summary-box", elem_classes="vie-scroll")
                tabs = {}
                for k, (name, _desc) in METHOD_INFO.items():
                    with gr.Tab(f"{k} {name}"):
                        info = gr.Markdown()
                        with gr.Row(equal_height=False):
                            vis = gr.Image(label="Vùng phát hiện", height="82vh", scale=3)
                            with gr.Column(scale=2):
                                fields = gr.HTML(fields_html(None) + LEGEND_HTML,
                                                 elem_classes=["vie-scroll", "fields-box"])
                                items = gr.HTML(items_html(None if k in "AC" else {"line_items": []}),
                                                elem_classes="vie-scroll")
                        gr.HTML(pipeline_html(k))
                        with gr.Accordion("JSON & trace", open=False):
                            with gr.Row():
                                js = gr.JSON(label="Canonical JSON")
                                trace = gr.Code(label="Trace (layout / OCR / raw output)", language=None)
                        tabs[k] = (info, vis, fields, items, js, trace)

        outputs = [summary, status] + [c for k in "ABCD" for c in tabs[k]]

        def run(img, selected, vlm_id, llm_id, max_side, low_mem):
            import gradio as gr  # noqa: F811

            if not img:
                raise gr.Error("Hãy tải ảnh lên trước.")
            if not selected:
                raise gr.Error("Hãy chọn ít nhất một phương pháp.")
            path = prepare_image(img, int(max_side))
            results = {}
            per = {k: [gr.update()] * 6 for k in "ABCD"}

            def pack(msg):
                flat = [c for k in "ABCD" for c in per[k]]
                return [summary_html(results), msg] + flat

            for k in [k for k in "ABCD" if k in selected]:
                per[k][0] = f"⏳ Đang chạy {k}…"
                yield pack(f"⏳ Đang chạy **{k}**, model đang nạp: {BACKENDS.status()}")
                try:
                    res = run_one(k, str(path), vlm_id, llm_id)
                    results[k] = res
                    doc = res["doc"]
                    info = (f"✅ Dự đoán **{res['t_pred']:.1f}s** (nạp/khởi tạo {res['t_load']:.1f}s)"
                            + ("  ⚠ model không trả JSON hợp lệ (parse fail)" if res["parse_fail"] else ""))
                    per[k] = [info, res["image"], fields_html(doc) + LEGEND_HTML, items_html(doc),
                              doc, res["trace"]]
                except Exception as exc:
                    tb = traceback.format_exc()
                    print(tb, file=sys.stderr)
                    results[k] = {"error": str(exc)}
                    per[k] = [f"❌ Lỗi: `{exc}`", None, fields_html(None) + LEGEND_HTML,
                              items_html(None), None, tb]
                if low_mem:
                    BACKENDS.unload()
            yield pack(f"Xong. Model đang nạp: {BACKENDS.status()}")

        def unload():
            BACKENDS.unload()
            return f"Model đang nạp: {BACKENDS.status()}"

        for g, paths in galleries:
            def pick(evt: gr.SelectData, paths=paths):
                return paths[evt.index]
            g.select(pick, None, img_in)

        run_btn.click(run, [img_in, methods_in, vlm_in, llm_in, max_side_in, low_mem_in], outputs)
        unload_btn.click(unload, None, status)
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--share", action="store_true", help="tao link public *.gradio.live")
    ap.add_argument("--device", choices=["auto", "cpu"], default="auto")
    ap.add_argument("--vlm", default=os.environ.get("VIE_VLM", "Qwen/Qwen3-VL-2B-Instruct"))
    ap.add_argument("--llm", default=os.environ.get("VIE_LLM", "Qwen/Qwen3-4B-Instruct-2507"))
    ap.add_argument("--max-new-tokens", type=int, default=None, help="ghi de VIE_MAX_NEW_TOKENS (mac dinh 768)")
    ap.add_argument("--max-side", type=int, default=2000, help="resize canh dai anh dau vao (0 = giu nguyen)")
    ap.add_argument("--low-mem", action="store_true", help="giai phong model sau moi phuong phap")
    args = ap.parse_args()

    configure_runtime(args.device, args.max_new_tokens)
    print(f"[demo] paddle={os.environ['VIE_PADDLE_DEVICE']} torch={os.environ['VIE_TORCH_DEVICE']} "
          f"dtype={os.environ['VIE_TORCH_DTYPE']}")
    print(f"[demo] layout={os.environ.get('VIE_LAYOUT_DIR')}  rec={os.environ.get('VIE_REC_DIR')}")
    ui = build_ui(args)
    ui.queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, share=args.share, css=CSS)


if __name__ == "__main__":
    main()
