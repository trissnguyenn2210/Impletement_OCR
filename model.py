"""PP-StructureV3 pipeline factory."""

import importlib.util
import os
import sys

from config import CONFIG, Config


def _allow_headless_opencv() -> None:
    """Treat headless OpenCV as valid for PaddleX's OCR dependency check."""
    from paddlex.utils import deps

    if importlib.util.find_spec("cv2") and not deps.is_dep_available(
        "opencv-contrib-python"
    ):
        cv2 = __import__("cv2")
        original = deps.is_dep_available

        def is_dep_available(dep: str, /, check_version: bool = False) -> bool:
            if dep == "opencv-contrib-python":
                return True
            return original(dep, check_version=check_version)

        deps.is_dep_available = is_dep_available
        deps.is_extra_available.cache_clear()

        # PaddleX may import this reader before the dependency shim is applied.
        for module in sys.modules.values():
            if getattr(module, "__name__", "").startswith("paddlex."):
                module.__dict__.setdefault("cv2", cv2)


def build_pipeline(config: Config = CONFIG):
    """Build the pipeline and use the fine-tuned layout model when available."""
    # BOS is usually easier to reach in server environments than Hugging Face.
    os.environ.setdefault("PADDLE_PDX_MODEL_SOURCE", "bos")
    _allow_headless_opencv()
    from paddleocr import PPStructureV3

    kwargs = {
        "layout_detection_model_name": config.model_name,
        "device": config.device,
        "use_doc_orientation_classify": config.use_doc_orientation,
        "use_doc_unwarping": config.use_doc_unwarping,
        "use_textline_orientation": config.use_textline_orientation,
        "use_formula_recognition": False,
        "use_chart_recognition": False,
    }

    if config.finetuned_layout_dir.exists():
        kwargs["layout_detection_model_dir"] = str(config.finetuned_layout_dir)

    return PPStructureV3(**kwargs)
