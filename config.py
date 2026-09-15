"""Project defaults. Runtime overrides are provided through argparse."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT.parent / "data" / "OCR_dataset_coco"
DEFAULT_RECOGNITION_DATA_DIR = (
    ROOT.parent
    / "data"
    / "OCR_dataset"
    / "text_recognition_mcocr_data"
    / "text_recognition_mcocr_data"
)
DEFAULT_RECOGNITION_TRAIN_LABELS = (
    ROOT.parent / "data" / "OCR_dataset" / "text_recognition_train_data.txt"
)
DEFAULT_RECOGNITION_VAL_LABELS = (
    ROOT.parent / "data" / "OCR_dataset" / "text_recognition_val_data.txt"
)


@dataclass(frozen=True)
class Config:
    project_dir: Path = ROOT
    dataset_dir: Path = DEFAULT_DATASET_DIR
    output_dir: Path = ROOT / "outputs" / "layout"
    paddlex_dir: Path = ROOT / "vendor" / "PaddleX"
    pretrain_path: Path = ROOT / "models" / "PP-DocLayout_plus-L_pretrained.pdparams"
    layout_config: Path = ROOT / "configs" / "PP-DocLayout_plus-L.yaml"
    labels_file: Path = ROOT / "configs" / "layout_labels.txt"
    pipeline_config: Path = ROOT / "configs" / "PP-StructureV3.yaml"
    recognition_output_dir: Path = ROOT / "outputs" / "recognition"
    recognition_config: Path = ROOT / "configs" / "PP-OCRv5_server_rec.yaml"
    recognition_data_dir: Path = DEFAULT_RECOGNITION_DATA_DIR
    recognition_train_labels: Path = DEFAULT_RECOGNITION_TRAIN_LABELS
    recognition_val_labels: Path = DEFAULT_RECOGNITION_VAL_LABELS
    recognition_character_dict: Path = (
        ROOT / "outputs" / "recognition" / "character_dict.txt"
    )

    model_name: str = "PP-DocLayout_plus-L"
    recognition_model_name: str = "PP-OCRv5_server_rec"
    recognition_pretrained_model: str = (
        "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/"
        "PP-OCRv5_server_rec_pretrained.pdparams"
    )
    device: str = "gpu:0"
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 0.0001
    recognition_epochs: int = 20
    recognition_batch_size: int = 32
    recognition_num_workers: int = 4
    recognition_learning_rate: float = 0.0001
    recognition_max_text_length: int = 160
    recognition_eval_batch_step: int = 500
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0
    use_doc_orientation: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False

    @property
    def num_classes(self) -> int:
        labels = [
            line.strip()
            for line in self.labels_file.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if not labels:
            raise ValueError(f"No layout labels found in {self.labels_file}")
        return len(labels)

    @property
    def finetuned_layout_dir(self) -> Path:
        return self.output_dir / "best_model" / "inference"

    @property
    def finetuned_recognition_dir(self) -> Path:
        return self.recognition_output_dir / "inference"

    @property
    def paddlex_main(self) -> Path:
        return self.paddlex_dir / "main.py"

    @property
    def paddleocr_dir(self) -> Path:
        return self.paddlex_dir / "paddlex" / "repo_manager" / "repos" / "PaddleOCR"


CONFIG = Config()
