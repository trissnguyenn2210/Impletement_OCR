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
import math
import os
from pathlib import Path
import types

from .base import Method, register
from .postproc import (
    parse_json_loose,
    paddle_ocr_text_to_doc,
    regions_to_doc,
)

# ---------------------------------------------------------------- cau hinh
# Ghi de bang bien moi truong, khong phai sua code:
#   export VIE_VLM=Qwen/Qwen3-VL-4B-Instruct
#   export VIE_LLM=Qwen/Qwen3-4B-Instruct-2507
#   export VIE_LAYOUT_DIR=/path/den/inference      (layout da fine-tune)
#   export VIE_DET_DIR=/path/den/inference         (PP-OCRv5 det)
#   export VIE_REC_DIR=/path/den/inference         (rec da fine-tune)
#   export VIE_ATTN_IMPL=sdpa                     (Qwen3-VL: sdpa/eager/flash_attention_2)
#   export VIE_LLM_ATTN_IMPL=sdpa                 (text LLM D: sdpa/eager/flash_attention_2)
#   export VIE_INTERNVL_ATTN_IMPL=sdpa            (InternVL: sdpa/eager/flash_attention_2)
#   export VIE_INTERNVL_MAX_BATCH_SIZE=2          (InternVL: auto-split + OOM retry)
#   export VIE_INTERNVL_MAX_NEW_TOKENS=384       (InternVL decode budget)
#   export VIE_QWEN_MAX_PIXELS=1003520            (tuy chon, giam vision tokens)
#   export VIE_QWEN_MAX_BATCH_SIZE=2              (tuy chon, bat batch Qwen)
#   export VIE_QWEN8_MULTI_GPU=1                  (Qwen8 shard across visible GPUs)
VLM_ID = os.environ.get("VIE_VLM", "Qwen/Qwen3-VL-4B-Instruct")
LLM_ID = os.environ.get("VIE_LLM", "Qwen/Qwen3-4B-Instruct-2507")
# Keep the two runtimes' device syntax separate. PaddleX accepts ``gpu:0``;
# Torch/Transformers accepts ``cuda:0``.  A and the Paddle part of C/D use
# PADDLE_DEVICE, while B and the VLM/LLM part of C/D use TORCH_DEVICE.
# Override for machines without a CUDA build, e.g. VIE_PADDLE_DEVICE=cpu,
# VIE_TORCH_DEVICE=cpu (the web demo sets these automatically).
PADDLE_DEVICE = os.environ.get("VIE_PADDLE_DEVICE", "gpu:0")
TORCH_DEVICE = os.environ.get("VIE_TORCH_DEVICE", "cuda:0")
# Keep the general evaluator's historical budget.  The optimized InternVL
# Slurm jobs explicitly set VIE_MAX_NEW_TOKENS=384; other methods (especially
# the text-only LLM path) retain the safer 768-token default.
MAX_NEW = int(os.environ.get("VIE_MAX_NEW_TOKENS", "768"))


def _torch_dtype():
    """BF16 by default; VIE_TORCH_DTYPE=float16/float32 for GPUs without BF16
    (e.g. Turing GTX/RTX 20xx) or for CPU runs."""
    import torch

    name = os.environ.get("VIE_TORCH_DTYPE", "bfloat16").strip().lower()
    table = {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
             "float16": torch.float16, "fp16": torch.float16, "half": torch.float16,
             "float32": torch.float32, "fp32": torch.float32}
    if name not in table:
        raise ValueError(f"VIE_TORCH_DTYPE khong hop le: {name!r}")
    return table[name]


def _patch_paddle_inference_compat():
    """Bridge PaddleX 3.4 to Paddle 2.6's renamed inference API.

    The environment used by this checkout has Paddle 2.6.2 while the
    installed PaddleX calls ``AnalysisConfig.set_optimization_level``.  The
    equivalent Paddle 2.6 switch is ``switch_ir_optim``.  Keep this shim
    local to inference startup instead of modifying the installed package.
    """
    import paddle

    analysis_config = paddle.base.libpaddle.AnalysisConfig
    if hasattr(analysis_config, "set_optimization_level"):
        return

    def set_optimization_level(self, _level):
        if hasattr(self, "switch_ir_optim"):
            self.switch_ir_optim(True)

    setattr(analysis_config, "set_optimization_level", set_optimization_level)


def _paddle_kw(model_name):
    """Constructor kwargs chung cho cac predictor PaddleOCR.

    Tren CPU, Paddle 3.x + oneDNN loi khi chay graph PIR cua cac model nay:
    "ConvertPirAttribute2RuntimeAttribute not support ArrayAttribute".
    Mac dinh tat MKLDNN tren CPU; bat lai bang VIE_PADDLE_MKLDNN=1.
    """
    kw = {"model_name": model_name, "device": PADDLE_DEVICE}
    if PADDLE_DEVICE.startswith("cpu") and os.environ.get(
            "VIE_PADDLE_MKLDNN", "0").strip().lower() not in {"1", "true", "yes", "on"}:
        kw["enable_mkldnn"] = False
    return kw


def _positive_int_env(name, default):
    """Read a positive integer tuning knob from the environment."""
    raw = os.environ.get(name, str(default))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} phai la so nguyen duong, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name} phai >= 1, got {value}")
    return value


def _qwen_attention_impl():
    """Resolve the Qwen3-VL attention backend without hiding bad settings."""
    value = os.environ.get("VIE_ATTN_IMPL", "sdpa").strip().lower()
    if value in ("", "auto"):
        value = "sdpa"
    allowed = {"eager", "sdpa", "flash_attention_2"}
    if value not in allowed:
        raise ValueError(
            "VIE_ATTN_IMPL phai la eager, sdpa hoac flash_attention_2; "
            f"got {value!r}"
        )
    if value == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "VIE_ATTN_IMPL=flash_attention_2 nhung flash-attn chua duoc "
                "cai. Dung VIE_ATTN_IMPL=sdpa hoac cai wheel phu hop CUDA/PyTorch."
            ) from exc
    return value


def _internvl_attention_impl():
    """Resolve InternVL attention without requiring flash-attn wheels.

    InternVL's remote code hard-codes eager attention when FlashAttention is
    unavailable.  PyTorch SDPA is a better default for the Blackwell-targeted
    torch build, so the wrapper enables it after loading the model and keeps
    FlashAttention opt-in because its wheel must match the exact torch/CUDA
    pair.
    """
    value = os.environ.get("VIE_INTERNVL_ATTN_IMPL", "sdpa").strip().lower()
    if value in ("", "auto"):
        value = "sdpa"
    allowed = {"eager", "sdpa", "flash_attention_2"}
    if value not in allowed:
        raise ValueError(
            "VIE_INTERNVL_ATTN_IMPL phai la eager, sdpa hoac flash_attention_2; "
            f"got {value!r}"
        )
    if value == "flash_attention_2":
        try:
            import flash_attn  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "VIE_INTERNVL_ATTN_IMPL=flash_attention_2 nhung flash-attn "
                "chua duoc cai. Dung sdpa hoac cai wheel phu hop CUDA/PyTorch."
            ) from exc
    return value


def _internvl_sdpa_attention(module, hidden_states):
    """SDPA replacement for InternVL's custom vision attention.

    The remote vision implementation only offers a quadratic eager path when
    flash-attn is missing.  This preserves its QK normalization and projection
    while delegating the attention kernel to PyTorch's fused SDPA dispatcher.
    """
    import torch.nn.functional as F

    batch, seq_len, hidden = hidden_states.shape
    qkv = module.qkv(hidden_states).reshape(
        batch, seq_len, 3, module.num_heads, hidden // module.num_heads
    ).permute(2, 0, 3, 1, 4)
    query, key, value = qkv.unbind(0)

    if module.qk_normalization:
        batch_, heads_, seq_, head_dim = query.shape
        query = module.q_norm(
            query.transpose(1, 2).flatten(-2, -1)
        ).view(batch_, seq_, heads_, head_dim).transpose(1, 2)
        key = module.k_norm(
            key.transpose(1, 2).flatten(-2, -1)
        ).view(batch_, seq_, heads_, head_dim).transpose(1, 2)

    attended = F.scaled_dot_product_attention(
        query,
        key,
        value,
        dropout_p=module.attn_drop.p if module.training else 0.0,
        is_causal=False,
    )
    output = attended.transpose(1, 2).reshape(batch, seq_len, hidden)
    output = module.proj(output)
    return module.proj_drop(output)

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
def _subtask_from_log(train_on):
    """Load sub-task metrics from the explicit file or training logs."""
    from subtask_metrics import load_subtask_metrics

    return load_subtask_metrics(train_on)


def _cost_from_log(key):
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


def _is_usable_inference_dir(path):
    """Ignore Git-LFS pointer files when selecting a local checkpoint."""
    path = Path(path)
    params = path / "inference.pdiparams"
    config = path / "inference.yml"
    if not params.is_file() or not config.is_file() or params.stat().st_size <= 1024:
        return False
    try:
        with params.open("rb") as stream:
            header = stream.read(64)
    except OSError:
        return False
    return not header.startswith(b"version https://git-lfs.github.com/spec/v1")


def _model_dir(env, defaults):
    """Resolve an override, then outputs/, then the local checkpoints/."""
    candidates = []
    override = os.environ.get(env)
    if override:
        candidates.append(Path(override).expanduser())
    candidates.extend(defaults)
    for candidate in candidates:
        if _is_usable_inference_dir(candidate):
            return str(candidate)
    return None


# ===================================================================== backends
class _Layout:
    """Layout detector. Tra ve [{'label','box','score'}].

    Dung model da fine-tune neu co (outputs/layout/best_model/inference),
    nguoc lai dung PP-DocLayout_plus-L goc.
    """

    def __init__(self, train_on=None):
        import cfg
        _patch_paddle_inference_compat()
        from paddleocr import LayoutDetection

        default = cfg.ROOT / "outputs" / "layout" / "best_model" / "inference"
        mdir = _model_dir(
            "VIE_LAYOUT_DIR",
            (default, cfg.ROOT / "checkpoints" / "layout" / "inference"),
        )
        kw = _paddle_kw("PP-DocLayout_plus-L")
        if mdir:
            kw["model_dir"] = mdir
        print(f"   [Layout] model_dir={mdir or '(goc, chua fine-tune)'}")
        self.m = LayoutDetection(**kw)

    def __call__(self, img_path):
        res = self.m.predict(str(img_path), batch_size=1)
        return self._parse_result(res)

    @staticmethod
    def _parse_result(res):
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

    def batch(self, image_paths, batch_size):
        """Run layout detection for several full pages in one predictor."""
        if not image_paths:
            return []
        sources = [str(path) for path in image_paths]
        results = self.m.predict(sources, batch_size=batch_size)
        if len(results) != len(sources):
            raise RuntimeError(
                f"layout predict tra {len(results)} output cho {len(sources)} anh"
            )
        return [self._parse_result([result]) for result in results]


class _OCR:
    """Text detection + CTC recognition. Doc chu trong mot anh hoac mot crop."""

    def __init__(self, train_on=None):
        import cfg
        _patch_paddle_inference_compat()
        from paddleocr import TextDetection, TextRecognition

        rec_dir = _model_dir(
            "VIE_REC_DIR",
            (
                cfg.ROOT / "outputs" / "recognition" / "inference",
                cfg.ROOT / "checkpoints" / "recognition" / "inference",
            ),
        )
        det_dir = _model_dir("VIE_DET_DIR", ())
        det_kw = _paddle_kw("PP-OCRv5_server_det")
        if det_dir:
            det_kw["model_dir"] = det_dir
        self.det = TextDetection(**det_kw)
        kw = _paddle_kw("PP-OCRv5_server_rec")
        if rec_dir:
            kw["model_dir"] = rec_dir
        print(
            f"   [Det] model_dir={det_dir or '(goc, chua fine-tune)'}\n"
            f"   [Rec] model_dir={rec_dir or '(goc, chua fine-tune)'}"
        )
        self.rec = TextRecognition(**kw)
        # PP-OCRv5_server_det keeps the aspect ratio of each crop before its
        # ToBatch step; arbitrary layout crops therefore cannot be stacked
        # safely.  Leave this at 1 unless a caller has pre-bucketed equal-size
        # inputs.  Recognition has fixed preprocessing and can be batched.
        self.det_batch_size = _positive_int_env("VIE_A_DET_BATCH", 1)
        # Keep the historical per-layout-crop recognition batch.  The CTC
        # decoder chooses its width ratio from the current batch, so merging
        # line crops from different layout regions can change decoded text.
        self.rec_batch_size = _positive_int_env("VIE_A_REC_BATCH", 16)

    def lines(self, image):
        """image = duong dan hoac numpy array. Tra [{'text','box'}] theo thu tu y."""
        return self.lines_batch([image], det_batch=1)[0]

    def lines_batch(self, images, det_batch=None,
                    rec_batch=None) :
        """OCR several layout crops with shared detection/recognition batches.

        The old implementation called text detection once per crop and only
        batched recognition inside that crop.  A layout page often has many
        crops, so the expensive predictor launches were repeated hundreds of
        times.  This method flattens all crops from the current document batch,
        runs detection once per Paddle batch, then recognizes all line crops in
        larger batches while retaining each crop's original owner.
        """
        import numpy as np

        if not images:
            return []
        det_batch = det_batch or self.det_batch_size
        rec_batch = rec_batch or self.rec_batch_size
        arrays = [self._as_array(image) for image in images]
        sources = [
            str(image) if isinstance(image, (str, Path)) else array
            for image, array in zip(images, arrays)
        ]
        det_results = self.det.predict(sources, batch_size=det_batch)
        if len(det_results) != len(images):
            raise RuntimeError(
                f"text detection tra {len(det_results)} output cho {len(images)} crop"
            )

        answers = [[] for _ in images]
        line_crops = []
        line_refs = []
        for image_index, (img, result) in enumerate(zip(arrays, det_results)):
            polys = result.get("dt_polys")
            if polys is None:
                polys = []
            for poly in polys:
                p = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                x1, y1 = p.min(axis=0)
                x2, y2 = p.max(axis=0)
                x1, y1 = max(0, int(x1)), max(0, int(y1))
                x2 = min(img.shape[1], int(x2))
                y2 = min(img.shape[0], int(y2))
                if x2 - x1 < 3 or y2 - y1 < 3:
                    continue
                line_crops.append(img[y1:y2, x1:x2])
                line_refs.append((image_index, [x1, y1, x2, y2]))

        if not line_crops:
            return answers

        rec_results = self.rec.predict(line_crops, batch_size=rec_batch)
        if len(rec_results) != len(line_refs):
            raise RuntimeError(
                f"text recognition tra {len(rec_results)} output cho "
                f"{len(line_refs)} dong"
            )
        for result, (image_index, box) in zip(rec_results, line_refs):
            text = str(result.get("rec_text", ""))
            if text.strip():
                answers[image_index].append({"text": text, "box": box})
        return [sorted(lines, key=lambda d: (d["box"][1], d["box"][0]))
                for lines in answers]

    @staticmethod
    def _as_array(image):
        import cv2
        import numpy as np
        from PIL import Image

        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise RuntimeError(f"khong doc duoc anh {image}")
            return img
        if isinstance(image, Image.Image):
            # _crop() returns RGB PIL images; PaddleOCR/OpenCV convention is BGR.
            return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        return np.asarray(image)


def _patch_paddleocr_transformers_compat():
    """Bridge PaddleOCR-VL remote code to the installed Transformers API.

    The PaddleOCR-VL revision currently cached on Hugging Face calls
    ``create_causal_mask(inputs_embeds=...)``.  Transformers 4.57 renamed
    that keyword to ``input_embeds``.  Newer Transformers revisions also
    removed the ``default`` entry from ``ROPE_INIT_FUNCTIONS`` even though
    the PaddleOCR-VL remote code still requests it.  Patch both interfaces
    before ``trust_remote_code`` imports the model module; the wrappers
    preserve the current API for all other callers in this process.
    """
    import inspect
    import torch
    import transformers.masking_utils as masking_utils
    import transformers.modeling_rope_utils as rope_utils

    # Torch 2.14 registers a Triton ``bmm`` override for this exact outer
    # product shape.  The container has no Python development headers, so
    # Triton cannot JIT its helper even though ordinary CUDA bmm works.  The
    # PaddleOCR-VL RoPE path only needs the standard CUDA implementation.
    native_registry = getattr(getattr(torch, "_native", None), "registry", None)
    disable_overrides = getattr(native_registry, "deregister_op_overrides", None)
    if disable_overrides is not None:
        disable_overrides(disable_op_symbols="bmm")

    if "default" not in rope_utils.ROPE_INIT_FUNCTIONS:
        def compute_default_rope_parameters(
            config=None, device=None, seq_len=None, layer_type=None
        ):
            """Compatibility implementation of the standard RoPE formula."""
            import torch

            if config is None:
                raise ValueError("PaddleOCR-VL default RoPE requires a config")
            rope_scaling = getattr(config, "rope_scaling", None) or {}
            base = rope_scaling.get("rope_theta", getattr(config, "rope_theta", 10000.0))
            head_dim = getattr(config, "head_dim", None)
            if head_dim is None:
                head_dim = config.hidden_size // config.num_attention_heads
            partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base
                ** (
                    torch.arange(0, dim, 2, dtype=torch.int64)
                    .to(device=device, dtype=torch.float)
                    / dim
                )
            )
            return inv_freq, 1.0

        rope_utils.ROPE_INIT_FUNCTIONS["default"] = compute_default_rope_parameters

    original = masking_utils.create_causal_mask
    params = inspect.signature(original).parameters
    if getattr(original, "_vie_paddle_compat", False):
        return

    needs_inputs_alias = "inputs_embeds" not in params and "input_embeds" in params
    needs_cache_drop = "cache_position" not in params
    if not needs_inputs_alias and not needs_cache_drop:
        return

    def create_causal_mask_compat(*args, **kwargs):
        if needs_inputs_alias and "inputs_embeds" in kwargs and "input_embeds" not in kwargs:
            kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
        if needs_cache_drop:
            kwargs.pop("cache_position", None)
        return original(*args, **kwargs)

    create_causal_mask_compat.__name__ = getattr(
        original, "__name__", "create_causal_mask"
    )
    create_causal_mask_compat._vie_paddle_compat = True
    masking_utils.create_causal_mask = create_causal_mask_compat


def _patch_paddleocr_generation_compat(model):
    """Restore the cache-position kwarg removed by Transformers 5.x.

    The PaddleOCR-VL remote ``prepare_inputs_for_generation`` still uses
    ``cache_position[0]`` to drop image tensors after the prefill step.  The
    current Transformers generation loop no longer synthesizes that kwarg for
    remote models whose outer ``forward`` only exposes it through ``**kwargs``.
    Compute the same range immediately before calling the remote method.
    """
    original = model.prepare_inputs_for_generation
    if getattr(original, "_vie_paddle_cache_compat", False):
        return

    def prepare_inputs_compat(self, input_ids, *args, **kwargs):
        if kwargs.get("cache_position") is None:
            import torch

            past_key_values = kwargs.get("past_key_values")
            if past_key_values is None:
                past_seen_tokens = 0
            elif hasattr(past_key_values, "get_seq_length"):
                past_seen_tokens = past_key_values.get_seq_length()
            else:
                # Fallback for legacy tuple caches; Transformers 5 uses
                # DynamicCache, but this keeps the bridge harmless on older
                # supported versions.
                past_seen_tokens = past_key_values[0][0].shape[-2]
            sequence_length = kwargs.get("next_sequence_length")
            if sequence_length is None:
                sequence_length = input_ids.shape[-1]
            kwargs["cache_position"] = (
                torch.arange(sequence_length, device=input_ids.device)
                + past_seen_tokens
            )
        return original(input_ids, *args, **kwargs)

    prepare_inputs_compat._vie_paddle_cache_compat = True
    model.prepare_inputs_for_generation = types.MethodType(
        prepare_inputs_compat, model
    )


class _VLM:
    """VLM qua transformers. Dung cho B (ca trang) va C (tung crop)."""

    def __init__(self, model_id=VLM_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor

        print(f"   [VLM] {model_id} on {TORCH_DEVICE}")
        model_name = Path(model_id.rstrip("/")).name.lower()
        self.is_paddleocr_vl = (
            model_id.rstrip("/") == "PaddlePaddle/PaddleOCR-VL"
            or "paddleocr-vl" in model_name
        )
        self.is_internvl = "internvl" in model_id.lower()
        self.is_qwen3vl = "qwen3-vl" in model_id.lower()
        self.internvl_attention_impl = (
            _internvl_attention_impl() if self.is_internvl else None
        )
        if self.is_paddleocr_vl:
            _patch_paddleocr_transformers_compat()
        processor_kwargs = {"trust_remote_code": True}
        if self.is_internvl:
            # InternVL's tokenizer config predates Transformers' dynamic
            # multimodal-token support.  The processor expects these attrs
            # on Qwen2TokenizerFast, so provide the token mapping explicitly.
            processor_kwargs["extra_special_tokens"] = {
                "start_image_token": "<img>",
                "end_image_token": "</img>",
                "context_image_token": "<IMG_CONTEXT>",
                "video_token": "<|video_pad|>",
            }
        self.attn_implementation = None
        self.qwen_image_kwargs = {}
        if self.is_qwen3vl:
            self.attn_implementation = _qwen_attention_impl()
            # Keep the quality-preserving default unchanged.  These are
            # opt-in because a lower image budget trades OCR accuracy for
            # speed/memory.
            for env_name, kw_name in (
                ("VIE_QWEN_MIN_PIXELS", "min_pixels"),
                ("VIE_QWEN_MAX_PIXELS", "max_pixels"),
            ):
                if os.environ.get(env_name):
                    value = _positive_int_env(env_name, 1)
                    processor_kwargs[kw_name] = value
                    self.qwen_image_kwargs[kw_name] = value
        self.proc = AutoProcessor.from_pretrained(model_id, **processor_kwargs)
        if self.qwen_image_kwargs:
            # Qwen3VL's processor needs both bounds at call time.  If only
            # one knob was supplied, inherit the other bound from the model's
            # processor config instead of silently ignoring the override.
            size = getattr(getattr(self.proc, "image_processor", None), "size", {})
            self.qwen_image_kwargs.setdefault(
                "min_pixels", int(size.get("shortest_edge", 65536))
            )
            self.qwen_image_kwargs.setdefault(
                "max_pixels", int(size.get("longest_edge", 16777216))
            )
        # PaddleOCR-VL and InternVL expose their custom implementation through
        # AutoModelForCausalLM.  Qwen3-VL uses the image-text auto class.
        model_cls = (AutoModelForCausalLM
                     if self.is_paddleocr_vl or self.is_internvl
                     else AutoModelForImageTextToText)
        device_map = TORCH_DEVICE
        max_memory = None
        if self.is_qwen3vl and "qwen3-vl-8b" in model_id.lower() and os.environ.get(
                "VIE_QWEN8_CPU_OFFLOAD", "0").strip().lower() in {
                    "1", "true", "yes", "on"
                }:
            # A shared A100 may have less than the ~17 GiB required by the
            # BF16 Qwen8B weights.  Let Accelerate keep the vision/language
            # layers that fit on GPU and offload the remainder to CPU.
            device_map = "auto"
            max_memory = {
                0: os.environ.get("VIE_QWEN8_GPU_MEMORY", "12GiB"),
                "cpu": os.environ.get("VIE_QWEN8_CPU_MEMORY", "20GiB"),
            }
        if self.is_qwen3vl and "qwen3-vl-8b" in model_id.lower() and os.environ.get(
                "VIE_QWEN8_CPU_ONLY", "0").strip().lower() in {
                    "1", "true", "yes", "on"
                }:
            device_map = {"": "cpu"}
            max_memory = None
        if self.is_qwen3vl and "qwen3-vl-8b" in model_id.lower() and os.environ.get(
                "VIE_QWEN8_MULTI_GPU", "0").strip().lower() in {
                    "1", "true", "yes", "on"
                }:
            # A Qwen8B BF16 checkpoint is larger than one 10 GiB MIG slice,
            # but it fits when Accelerate shards it across two visible slices.
            # Keep a little headroom on each slice for vision activations and
            # allow CPU spillover for a few embedding/output layers.
            visible_gpu_count = torch.cuda.device_count()
            if visible_gpu_count < 2:
                raise RuntimeError(
                    "VIE_QWEN8_MULTI_GPU can be used only with at least two visible GPUs"
                )
            per_gpu_memory = os.environ.get("VIE_QWEN8_GPU_MEMORY", "8GiB")
            max_memory = {
                index: per_gpu_memory for index in range(visible_gpu_count)
            }
            max_memory["cpu"] = os.environ.get("VIE_QWEN8_CPU_MEMORY", "24GiB")
            device_map = "auto"
        if self.is_internvl and os.environ.get(
                "VIE_INTERNVL_CPU_ONLY", "0").strip().lower() in {
                    "1", "true", "yes", "on"
                }:
            # The shared GPU can be fully occupied by unrelated jobs.  Keep
            # an explicit CPU fallback so the small InternVL demo remains
            # runnable without evicting another process from the accelerator.
            device_map = {"": "cpu"}
            max_memory = None
        model_kwargs = {
            "torch_dtype": _torch_dtype(),
            "device_map": device_map,
            "trust_remote_code": True,
            # Accelerate is already a project dependency; this lowers CPU
            # peak memory while loading the four Qwen8B safetensor shards.
            "low_cpu_mem_usage": True,
        }
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        if self.is_internvl:
            # InternVL3.5 remote code predates Transformers 5.20 and skips
            # PreTrainedModel.post_init(), while the new loading path reads
            # ``all_tied_weights_keys`` before it returns the model.  The
            # older model has no tied-parameter map, so an empty class-level
            # fallback is the correct compatibility value during loading.
            from transformers import PreTrainedModel

            if not hasattr(PreTrainedModel, "all_tied_weights_keys"):
                PreTrainedModel.all_tied_weights_keys = {}
        if self.is_paddleocr_vl:
            # PaddleOCR-VL's cached config defaults to eager attention and
            # use_cache=False.  That is safe but unnecessarily slow for the
            # long autoregressive JSON decode used by this evaluator.
            paddle_attn = os.environ.get("VIE_PADDLE_ATTN_IMPL", "sdpa").strip().lower()
            if paddle_attn not in {"eager", "sdpa", "flash_attention_2"}:
                raise ValueError(
                    "VIE_PADDLE_ATTN_IMPL phai la eager, sdpa hoac flash_attention_2; "
                    f"got {paddle_attn!r}"
                )
            if paddle_attn == "flash_attention_2":
                try:
                    import flash_attn  # noqa: F401
                except ImportError as exc:
                    raise RuntimeError(
                        "VIE_PADDLE_ATTN_IMPL=flash_attention_2 nhung flash-attn chua cai"
                    ) from exc
            model_kwargs["attn_implementation"] = paddle_attn
        if self.is_internvl:
            # The remote InternVL constructor accepts this flag and otherwise
            # chooses FlashAttention when the optional module happens to be
            # installed.  Make the requested backend deterministic.
            model_kwargs["use_flash_attn"] = (
                self.internvl_attention_impl == "flash_attention_2"
            )
        if self.attn_implementation is not None:
            model_kwargs["attn_implementation"] = self.attn_implementation
        self.model = model_cls.from_pretrained(
            model_id,
            **model_kwargs,
        ).eval()
        if self.is_paddleocr_vl:
            _patch_paddleocr_generation_compat(self.model)
        self.paddle_use_cache = (
            os.environ.get("VIE_PADDLE_USE_CACHE", "1").strip().lower()
            not in {"0", "false", "no", "off"}
        )
        if self.is_paddleocr_vl and self.paddle_use_cache:
            # Pass use_cache explicitly to generate() below as well; setting
            # both configs keeps Transformers' generation defaults aligned.
            self.model.config.use_cache = True
            generation_config = getattr(self.model, "generation_config", None)
            if generation_config is not None:
                generation_config.use_cache = True
        tf32_enabled = os.environ.get("VIE_TF32", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        if torch.cuda.is_available() and tf32_enabled:
            # A100 has fast TF32 paths for the remaining float32 matmuls.  The
            # main Qwen weights stay BF16, so this does not alter their dtype.
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        if self.is_qwen3vl:
            print(
                f"   [VLM] attention={self.attn_implementation} "
                f"qwen_max_batch={_positive_int_env('VIE_QWEN_MAX_BATCH_SIZE', 2)}"
            )
        if self.is_internvl:
            self._configure_internvl_attention()
            print(
                f"   [VLM] internvl_attention={self.internvl_attention_impl} "
                f"internvl_max_batch={_positive_int_env('VIE_INTERNVL_MAX_BATCH_SIZE', 2)}"
            )
            # The custom InternVL generate() API expects this runtime field;
            # the legacy model code normally initializes it inside chat().
            self.model.img_context_token_id = self.proc.image_token_id
        self.torch = torch

    def _configure_internvl_attention(self):
        """Use fused PyTorch attention when FlashAttention is not selected."""
        if self.internvl_attention_impl != "sdpa":
            return

        language_model = getattr(self.model, "language_model", None)
        setter = getattr(language_model, "set_attn_implementation", None)
        if setter is None:
            raise RuntimeError(
                "InternVL language model khong ho tro set_attn_implementation('sdpa')"
            )
        setter("sdpa")

        patched = 0
        vision_model = getattr(self.model, "vision_model", None)
        for module in vision_model.modules():
            if module.__class__.__name__ != "InternAttention":
                continue
            # Force the remote module through _naive_attn, which is replaced
            # by the SDPA implementation above.  This also handles the case
            # where flash-attn happens to be installed but SDPA was requested.
            module.use_flash_attn = False
            module._naive_attn = types.MethodType(
                _internvl_sdpa_attention, module
            )
            patched += 1
        if patched == 0:
            raise RuntimeError("Khong tim thay InternVL vision attention modules")

    @staticmethod
    def _as_rgb_image(image):
        from PIL import Image

        if isinstance(image, (str, Path)):
            with Image.open(image) as opened:
                return opened.convert("RGB")
        return image

    def _move_inputs(self, inputs):
        """Move tensor inputs to the model and match InternVL vision dtype."""
        if self.is_internvl:
            vision_dtype = next(self.model.vision_model.parameters()).dtype
        else:
            vision_dtype = None
        moved_inputs = {}
        for key, value in inputs.items():
            if not hasattr(value, "to"):
                moved_inputs[key] = value
                continue
            value = value.to(self.model.device)
            if (self.is_internvl and key == "pixel_values"
                    and value.is_floating_point()):
                value = value.to(dtype=vision_dtype)
            moved_inputs[key] = value
        return moved_inputs

    def _prompt_text(self, prompt):
        msgs = [{"role": "user", "content": [{"type": "image"},
                                                   {"type": "text", "text": prompt}]}]
        text = self.proc.apply_chat_template(msgs, tokenize=False,
                                              add_generation_prompt=True)
        if self.is_internvl:
            # The shipped template emits ``<image>`` while the processor
            # expands the InternVL ``<IMG_CONTEXT>`` token.
            text = text.replace("<image>", self.proc.image_token)
        return text

    def _default_max_new(self):
        if not self.is_internvl:
            return MAX_NEW
        if "VIE_INTERNVL_MAX_NEW_TOKENS" in os.environ:
            return _positive_int_env("VIE_INTERNVL_MAX_NEW_TOKENS", 384)
        if "VIE_MAX_NEW_TOKENS" in os.environ:
            return _positive_int_env("VIE_MAX_NEW_TOKENS", 384)
        return 384

    def ask(self, image, prompt, max_new=None):
        if max_new is None:
            max_new = self._default_max_new()
        image = self._as_rgb_image(image)
        if self.is_paddleocr_vl:
            # The standalone PaddleOCR-VL checkpoint is trained for the
            # element-level OCR task.  Its official Transformers prompt is
            # ``OCR:``; it is not an instruction-tuned invoice-JSON model.
            # The caller's JSON schema prompt would therefore produce empty
            # or non-JSON output and turn every image into parse_fail.
            prompt = "OCR:"
            msgs = [{"role": "user", "content": [{"type": "image", "image": image},
                                                       {"type": "text", "text": prompt}]}]
            inputs = self.proc.apply_chat_template(
                msgs,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            text = self._prompt_text(prompt)
            proc_kwargs = {}
            if self.qwen_image_kwargs:
                proc_kwargs["images_kwargs"] = self.qwen_image_kwargs
            inputs = self.proc(
                text=[text], images=[image], return_tensors="pt", **proc_kwargs
            )
        inputs = self._move_inputs(inputs)
        with self.torch.inference_mode():
            generation_kwargs = {
                "max_new_tokens": max_new,
                "do_sample": False,
            }
            if self.is_paddleocr_vl:
                generation_kwargs["use_cache"] = self.paddle_use_cache
            ids = self.model.generate(**inputs, **generation_kwargs)
        # InternVL's custom generate() passes only inputs_embeds to the
        # language model, so its returned ids contain generated tokens only;
        # the standard Transformers VLMs return prompt + continuation.
        generated_ids = ids if self.is_internvl else ids[:, inputs["input_ids"].shape[1]:]
        return self.proc.batch_decode(generated_ids,
                                      skip_special_tokens=True)[0].strip()

    def _internvl_batch_key(self, image):
        """Estimate exact dynamic patch count for padding-aware bucketing."""
        image = self._as_rgb_image(image)
        size = getattr(image, "size", None)
        if isinstance(size, tuple):
            width, height = size
        else:
            height, width = image.shape[:2]
        image_processor = self.proc.image_processor
        get_patches = getattr(image_processor, "get_number_of_image_patches", None)
        if get_patches is None:
            patch_count = 0
        else:
            patch_count = get_patches(
                height,
                width,
                {"crop_to_patches": True},
            )
        aspect = max(width, height) / max(1, min(width, height))
        # Patch count is the primary key because it controls image-token
        # padding; aspect ratio is the tie-breaker for more uniform tiling.
        return int(patch_count), round(math.log(aspect), 3), width * height

    def order_batch_indices(self, images):
        """Return a stable order that minimizes dynamic-image padding."""
        if not self.is_internvl:
            return list(range(len(images)))
        return sorted(
            range(len(images)),
            key=lambda index: self._internvl_batch_key(images[index]),
        )

    def _clear_cuda_cache(self):
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()

    def _is_cuda_oom(self, exc):
        oom_type = getattr(self.torch.cuda, "OutOfMemoryError", None)
        if oom_type is not None and isinstance(exc, oom_type):
            return True
        message = str(exc).lower()
        return "cuda out of memory" in message or "out of memory" in message

    def _ask_internvl_once(self, images, prompt, max_new):
        """Run one InternVL batch without retrying or changing its size."""
        texts = [self._prompt_text(prompt) for _ in images]
        tokenizer = self.proc.tokenizer
        old_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            inputs = self.proc(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
            )
            inputs = self._move_inputs(inputs)
            with self.torch.inference_mode():
                ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                )
            return [text.strip() for text in self.proc.batch_decode(
                ids, skip_special_tokens=True
            )]
        finally:
            tokenizer.padding_side = old_padding_side

    def _ask_internvl_resilient(self, images, prompt, max_new):
        """Retry an oversized batch with smaller chunks only after CUDA OOM."""
        try:
            return self._ask_internvl_once(images, prompt, max_new)
        except Exception as exc:
            if len(images) <= 1 or not self._is_cuda_oom(exc):
                raise
            self._clear_cuda_cache()
            midpoint = max(1, len(images) // 2)
            return (
                self._ask_internvl_resilient(images[:midpoint], prompt, max_new)
                + self._ask_internvl_resilient(images[midpoint:], prompt, max_new)
            )

    def ask_batch(self, images, prompt, max_new=None):
        """Generate responses for several images in one model call when safe.

        InternVL's processor concatenates dynamic image patches and returns a
        padded text batch; its custom ``generate`` implementation already
        supports the corresponding flattened vision features.  Qwen3-VL also
        supports padded image batches, but caps them with
        ``VIE_QWEN_MAX_BATCH_SIZE`` because image-token counts vary by receipt.
        """
        if max_new is None:
            max_new = self._default_max_new()
        images = [self._as_rgb_image(image) for image in images]
        if not images:
            return []
        if self.is_qwen3vl:
            if len(images) == 1:
                return [self.ask(images[0], prompt, max_new=max_new)]
            batch_cap = _positive_int_env("VIE_QWEN_MAX_BATCH_SIZE", 2)
            responses = []
            for start in range(0, len(images), batch_cap):
                chunk = images[start:start + batch_cap]
                if len(chunk) == 1:
                    responses.append(self.ask(chunk[0], prompt, max_new=max_new))
                else:
                    responses.extend(self._ask_qwen_batch(chunk, prompt, max_new))
            return responses
        if not self.is_internvl:
            return [self.ask(image, prompt, max_new=max_new) for image in images]

        order = self.order_batch_indices(images)
        ordered_images = [images[index] for index in order]
        batch_cap = _positive_int_env("VIE_INTERNVL_MAX_BATCH_SIZE", 2)
        ordered_responses = []
        for start in range(0, len(ordered_images), batch_cap):
            chunk = ordered_images[start:start + batch_cap]
            ordered_responses.extend(
                self._ask_internvl_resilient(chunk, prompt, max_new)
            )

        responses = [None] * len(images)
        for ordered_index, original_index in enumerate(order):
            responses[original_index] = ordered_responses[ordered_index]
        return responses

    def _ask_qwen_batch(self, images, prompt, max_new):
        """Run one padded Qwen3-VL generation call for a small image batch."""
        texts = [self._prompt_text(prompt) for _ in images]
        tokenizer = getattr(self.proc, "tokenizer", None)
        old_padding_side = getattr(tokenizer, "padding_side", None)
        if tokenizer is not None:
            # Left padding makes the generated continuation start at one
            # common sequence offset even when image prompts have different
            # token counts.
            tokenizer.padding_side = "left"
        try:
            proc_kwargs = {}
            if self.qwen_image_kwargs:
                proc_kwargs["images_kwargs"] = self.qwen_image_kwargs
            inputs = self.proc(
                text=texts,
                images=images,
                padding=True,
                return_tensors="pt",
                **proc_kwargs,
            )
            inputs = self._move_inputs(inputs)
            with self.torch.inference_mode():
                ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new,
                    do_sample=False,
                )
            cut = inputs["input_ids"].shape[1]
            return [text.strip() for text in self.proc.batch_decode(
                ids[:, cut:], skip_special_tokens=True
            )]
        finally:
            if tokenizer is not None and old_padding_side is not None:
                tokenizer.padding_side = old_padding_side


class _LLM:
    """LLM chi doc TEXT (khong thay anh). Dung cho D."""

    def __init__(self, model_id=LLM_ID):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"   [LLM] {model_id} on {TORCH_DEVICE}")
        self.tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        llm_attn = os.environ.get("VIE_LLM_ATTN_IMPL", "sdpa").strip().lower()
        if llm_attn in ("", "auto"):
            llm_attn = "sdpa"
        if llm_attn not in {"eager", "sdpa", "flash_attention_2"}:
            raise ValueError(
                "VIE_LLM_ATTN_IMPL phai la eager, sdpa hoac flash_attention_2; "
                f"got {llm_attn!r}"
            )
        if llm_attn == "flash_attention_2":
            try:
                import flash_attn  # noqa: F401
            except ImportError as exc:
                raise RuntimeError(
                    "VIE_LLM_ATTN_IMPL=flash_attention_2 nhung flash-attn chua cai"
                ) from exc
        print(f"   [LLM] attention={llm_attn}")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=_torch_dtype(), device_map=TORCH_DEVICE,
            trust_remote_code=True, low_cpu_mem_usage=True,
            attn_implementation=llm_attn).eval()
        self.model.config.use_cache = True
        generation_config = getattr(self.model, "generation_config", None)
        if generation_config is not None:
            generation_config.use_cache = True
        tf32_enabled = os.environ.get("VIE_TF32", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        if torch.cuda.is_available() and tf32_enabled:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.torch = torch

    def ask(self, prompt, max_new=MAX_NEW):
        msgs = [{"role": "user", "content": prompt}]
        text = self.tok.apply_chat_template(msgs, tokenize=False,
                                            add_generation_prompt=True)
        inputs = self.tok([text], return_tensors="pt").to(self.model.device)
        with self.torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=max_new,
                                      do_sample=False, use_cache=True)
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
        # The evaluator's batch-size controls how many full pages are sent to
        # this method.  Keep layout batches capped independently because the
        # large PP-DocLayout model is the first memory pressure point on a
        # 2g.10gb MIG slice.
        self.layout_batch_size = _positive_int_env(
            "VIE_A_LAYOUT_BATCH",
            min(_positive_int_env("VIE_INFER_BATCH_SIZE", 1), 4),
        )

    def predict(self, image_path):
        return self.predict_batch([image_path])[0]

    def predict_batch(self, image_paths):
        if not image_paths:
            return []

        regions_by_image = self.layout.batch(
            image_paths, batch_size=min(self.layout_batch_size, len(image_paths))
        )
        crops = []
        crop_refs = []
        for image_index, (image_path, regions) in enumerate(
                zip(image_paths, regions_by_image)):
            for region in regions:
                crop = _crop(image_path, region["box"])
                if crop is not None:
                    crop_refs.append((image_index, region))
                    crops.append(crop)

        # Keep OCR grouping per layout crop so CTC width handling and decoded
        # text remain identical to the established single-image path.  The
        # safe speedup is batching the large layout detector above; recognition
        # already batches up to VIE_A_REC_BATCH lines within each crop.
        lines_by_crop = [self.ocr.lines(crop) for crop in crops]
        outputs = [[] for _ in image_paths]
        for lines, (image_index, region) in zip(lines_by_crop, crop_refs):
            text = " ".join(d["text"] for d in lines)
            outputs[image_index].append({
                "label": region["label"],
                "text": text,
                "box": region["box"],
            })

        docs = []
        for out in outputs:
            doc = regions_to_doc(out)
            doc["line_items"] = None        # A KHONG sinh duoc line_items
            docs.append(doc)
        self.last_debug_batch = [{"regions": out} for out in outputs]
        self.last_debug = self.last_debug_batch[0] if len(self.last_debug_batch) == 1 else {}
        return docs

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

    def _paddle_doc(self, raw, image_path):
        """Map official PaddleOCR-VL OCR text into the evaluator schema."""
        if os.environ.get("VIE_PADDLE_RAW_DIR") and isinstance(raw, str):
            raw_dir = Path(os.environ["VIE_PADDLE_RAW_DIR"])
            raw_dir.mkdir(parents=True, exist_ok=True)
            raw_dir.joinpath(f"{Path(image_path).stem}.txt").write_text(
                raw, encoding="utf-8"
            )
        # D1 has no item annotations; D2 does.  The empty list is a valid
        # prediction for D2 and avoids confusing a non-JSON OCR response with
        # a model/inference failure.
        has_items = self.kw.get("test_on") in {"d2_train", "d2_val"}
        return paddle_ocr_text_to_doc(raw, has_items=has_items)

    def predict(self, image_path):
        raw = self.vlm.ask(image_path, SCHEMA_PROMPT)
        self.last_debug = {"raw_text": raw}
        if self.vlm.is_paddleocr_vl:
            return self._paddle_doc(raw, image_path)
        d = parse_json_loose(raw)
        if d is None:
            return None                  # -> tinh vao parse_fail
        if d.get("line_items") is None:
            d["line_items"] = []         # B PHAI tra list, [] = "khong co mon"
        return d

    def predict_batch(self, image_paths):
        responses = self.vlm.ask_batch(image_paths, SCHEMA_PROMPT)
        self.last_debug_batch = [{"raw_text": raw} for raw in responses]
        self.last_debug = self.last_debug_batch[0] if len(self.last_debug_batch) == 1 else {}
        docs = []
        for image_path, raw in zip(image_paths, responses):
            if self.vlm.is_paddleocr_vl:
                docs.append(self._paddle_doc(raw, image_path))
                continue
            d = parse_json_loose(raw)
            if d is not None and d.get("line_items") is None:
                d["line_items"] = []
            docs.append(d)
        return docs

    def order_batch_indices(self, image_paths):
        """Expose VLM-specific bucketing to the evaluator."""
        return self.vlm.order_batch_indices(image_paths)

    def subtask(self):
        return {"layout_mAP50": None, "layout_mAP50_95": None,
                "rec_acc": None, "rec_ned": None}        # khong co thanh phan nao

    def cost(self):
        return _cost_from_log("B")


# ============================================= cac bien the VLM cho Bang 1
def _mk_vlm(reg_name, model_id):
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
        self.last_debug = {"regions": out}
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
        self.last_debug = {"ocr_lines": lines}
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
