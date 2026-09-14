"""Project defaults. Runtime overrides are provided through argparse."""

from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_DIR = ROOT.parent / "data" / "OCR_dataset_coco"


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

    model_name: str = "PP-DocLayout_plus-L"
    device: str = "gpu:0"
    epochs: int = 100
    batch_size: int = 8
    num_workers: int = 4
    learning_rate: float = 0.0001
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
    def paddlex_main(self) -> Path:
        return self.paddlex_dir / "main.py"


CONFIG = Config()
