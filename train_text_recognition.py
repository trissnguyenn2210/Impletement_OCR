"""Fine-tune and export the PP-OCRv5 text-recognition model.

The existing layout training uses PaddleX's high-level entrypoint. Recognition
uses PaddleOCR's vendored training entrypoints directly because they support
the CTC/NRTR recognition dataset format used by the prepared crop dataset.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tqdm import tqdm

from config import CONFIG


TRAIN_PROGRESS_RE = re.compile(
    r"epoch:\s*\[(?P<epoch>\d+)\s*/\s*(?P<epochs>\d+)\].*?"
    r"global_step:\s*(?P<step>\d+)",
    re.IGNORECASE,
)
LOSS_RE = re.compile(
    r"\bloss:\s*(?P<loss>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class RecognitionPaths:
    data_dir: Path
    train_labels: Path
    val_labels: Path
    output_dir: Path
    character_dict: Path
    inference_dir: Path
    config: Path
    paddleocr_dir: Path


@dataclass(frozen=True)
class LabelRecord:
    image_name: str
    text: str
    image_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("check_dataset", "train", "evaluate", "export"),
        help="Operation to run (can also be passed as --mode)",
    )
    parser.add_argument(
        "--mode",
        dest="mode_option",
        choices=("check_dataset", "train", "evaluate", "export"),
        help="Operation to run",
    )
    parser.add_argument("--data-dir", type=Path, default=CONFIG.recognition_data_dir)
    parser.add_argument(
        "--train-labels", type=Path, default=CONFIG.recognition_train_labels
    )
    parser.add_argument(
        "--val-labels", type=Path, default=CONFIG.recognition_val_labels
    )
    parser.add_argument(
        "--output-dir", type=Path, default=CONFIG.recognition_output_dir
    )
    parser.add_argument(
        "--character-dict",
        type=Path,
        default=None,
        help="Output dictionary path; defaults to <output-dir>/character_dict.txt",
    )
    parser.add_argument(
        "--inference-dir",
        type=Path,
        default=None,
        help="Canonical exported model directory; defaults to <output-dir>/inference",
    )
    parser.add_argument("--config", type=Path, default=CONFIG.recognition_config)
    parser.add_argument(
        "--paddleocr-dir", type=Path, default=CONFIG.paddleocr_dir
    )
    parser.add_argument(
        "--pretrained-model",
        default=CONFIG.recognition_pretrained_model,
        help="Local .pdparams file/prefix or a PaddleOCR model URL",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Checkpoint prefix/file to resume training from",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        help="Checkpoint prefix/file for evaluate/export modes",
    )
    parser.add_argument("--device", default=CONFIG.device)
    parser.add_argument(
        "--epochs", type=int, default=CONFIG.recognition_epochs
    )
    parser.add_argument(
        "--batch-size", type=int, default=CONFIG.recognition_batch_size
    )
    parser.add_argument(
        "--num-workers", type=int, default=CONFIG.recognition_num_workers
    )
    parser.add_argument(
        "--learning-rate", type=float, default=CONFIG.recognition_learning_rate
    )
    parser.add_argument(
        "--max-text-length", type=int, default=CONFIG.recognition_max_text_length
    )
    parser.add_argument(
        "--eval-batch-step",
        type=int,
        default=CONFIG.recognition_eval_batch_step,
        help="Run validation every N training batches",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a tqdm progress bar during recognition training",
    )
    args = parser.parse_args()

    if args.mode and args.mode_option and args.mode != args.mode_option:
        parser.error("positional mode and --mode must match")
    args.mode = args.mode_option or args.mode or "train"

    if args.epochs < 1:
        parser.error("--epochs must be >= 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.num_workers < 0:
        parser.error("--num-workers must be >= 0")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be > 0")
    if args.max_text_length < 1:
        parser.error("--max-text-length must be >= 1")
    if args.eval_batch_step < 1:
        parser.error("--eval-batch-step must be >= 1")
    if args.resume and args.mode != "train":
        parser.error("--resume is only valid in train mode")
    if args.weights and args.mode not in {"evaluate", "export"}:
        parser.error("--weights is only valid in evaluate/export mode")
    return args


def resolve_paths(args: argparse.Namespace) -> RecognitionPaths:
    output_dir = args.output_dir.expanduser().resolve()
    return RecognitionPaths(
        data_dir=args.data_dir.expanduser().resolve(),
        train_labels=args.train_labels.expanduser().resolve(),
        val_labels=args.val_labels.expanduser().resolve(),
        output_dir=output_dir,
        character_dict=(
            args.character_dict or output_dir / "character_dict.txt"
        ).expanduser().resolve(),
        inference_dir=(
            args.inference_dir or output_dir / "inference"
        ).expanduser().resolve(),
        config=args.config.expanduser().resolve(),
        paddleocr_dir=args.paddleocr_dir.expanduser().resolve(),
    )


def require_path(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def read_label_file(label_path: Path, data_dir: Path) -> list[LabelRecord]:
    require_path(label_path, "label file")
    records: list[LabelRecord] = []
    for line_number, raw_line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not raw_line:
            continue
        if "\t" not in raw_line:
            raise ValueError(
                f"{label_path}:{line_number} must use '<image>\\t<text>' format"
            )
        image_name, text = raw_line.split("\t", 1)
        if not image_name:
            raise ValueError(f"{label_path}:{line_number} has an empty image path")
        if not text:
            raise ValueError(f"{label_path}:{line_number} has empty text")

        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = data_dir / image_path
        if not image_path.is_file():
            raise FileNotFoundError(
                f"Image referenced by {label_path}:{line_number} not found: "
                f"{image_path}"
            )
        records.append(LabelRecord(image_name, text, image_path))

    if not records:
        raise ValueError(f"No labels found in {label_path}")
    return records


def prepare_dataset(paths: RecognitionPaths, max_text_length: int) -> int:
    require_path(paths.data_dir, "recognition image directory")
    train_records = read_label_file(paths.train_labels, paths.data_dir)
    val_records = read_label_file(paths.val_labels, paths.data_dir)
    all_records = train_records + val_records

    characters = sorted(
        {
            character
            for record in all_records
            for character in record.text
            if character != " "
        }
    )
    longest = max(len(record.text) for record in all_records)
    if longest > max_text_length:
        raise ValueError(
            f"Longest label has {longest} characters, but --max-text-length is "
            f"{max_text_length}; increase it to avoid truncation."
        )

    paths.character_dict.parent.mkdir(parents=True, exist_ok=True)
    dictionary_text = "".join(f"{character}\n" for character in characters)
    if not paths.character_dict.exists() or (
        paths.character_dict.read_text(encoding="utf-8") != dictionary_text
    ):
        paths.character_dict.write_text(dictionary_text, encoding="utf-8")

    print(f"Recognition data: {paths.data_dir}")
    print(f"  train: {len(train_records)} samples")
    print(f"  val:   {len(val_records)} samples")
    print(f"  chars: {len(characters)} (+ space handled by use_space_char)")
    print(f"  max label length: {longest}")
    print(f"  character dict: {paths.character_dict}")
    return len(train_records)


def yaml_value(value: object) -> str:
    """Encode a scalar/list so PaddleOCR's -o parser can safely read it."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return json.dumps(list(value), ensure_ascii=False)
    if isinstance(value, Path):
        value = str(value)
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def options_command(
    executable: Path,
    script: Path,
    config_path: Path,
    options: list[tuple[str, object]],
) -> list[str]:
    command = [str(executable), str(script), "-c", str(config_path), "-o"]
    command.extend(f"{key}={yaml_value(value)}" for key, value in options)
    return command


def common_options(
    args: argparse.Namespace, paths: RecognitionPaths
) -> list[tuple[str, object]]:
    use_gpu = args.device.lower().startswith(("gpu", "cuda"))
    return [
        ("Global.model_name", CONFIG.recognition_model_name),
        ("Global.use_gpu", use_gpu),
        ("Global.character_dict_path", paths.character_dict),
        ("Global.max_text_length", args.max_text_length),
        ("Global.use_space_char", True),
        ("Global.uniform_output_enabled", False),
        ("Global.export_with_pir", False),
        ("Global.save_model_dir", paths.output_dir),
        ("Optimizer.lr.learning_rate", args.learning_rate),
        ("Train.dataset.data_dir", paths.data_dir),
        ("Train.dataset.label_file_list", [str(paths.train_labels)]),
        ("Train.sampler.first_bs", args.batch_size),
        ("Train.loader.batch_size_per_card", args.batch_size),
        ("Train.loader.num_workers", args.num_workers),
        ("Eval.dataset.data_dir", paths.data_dir),
        ("Eval.dataset.label_file_list", [str(paths.val_labels)]),
        ("Eval.loader.batch_size_per_card", args.batch_size),
        ("Eval.loader.num_workers", args.num_workers),
    ]


def train_command(
    args: argparse.Namespace, paths: RecognitionPaths
) -> list[str]:
    options = common_options(args, paths)
    options.extend(
        [
            ("Global.epoch_num", args.epochs),
            ("Global.eval_batch_step", [0, args.eval_batch_step]),
            ("Global.pretrained_model", "" if args.resume else args.pretrained_model),
            (
                "Global.checkpoints",
                resolve_checkpoint_prefix(args.resume) if args.resume else "",
            ),
        ]
    )
    return options_command(
        Path(sys.executable),
        paths.paddleocr_dir / "tools" / "train.py",
        paths.config,
        options,
    )


def checkpoint_candidates(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_dir():
        return [
            path / "model.pdparams",
            path / "best_accuracy.pdparams",
            path / "latest.pdparams",
        ]
    if path.name.endswith(".pdparams"):
        return [path]
    return [Path(f"{path}.pdparams")]


def resolve_checkpoint_prefix(path: Path | None) -> str:
    if path is None:
        raise ValueError("A checkpoint is required for this operation")
    candidates = checkpoint_candidates(path)
    checkpoint = next((candidate for candidate in candidates if candidate.is_file()), None)
    if checkpoint is None:
        raise FileNotFoundError(
            "Checkpoint not found. Tried: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    return str(checkpoint.with_suffix(""))


def default_checkpoint(paths: RecognitionPaths) -> Path:
    for candidate in (
        paths.output_dir / "best_model" / "model.pdparams",
        paths.output_dir / "best_accuracy.pdparams",
        paths.output_dir / "latest.pdparams",
    ):
        if candidate.is_file():
            return candidate
    return paths.output_dir / "best_model" / "model.pdparams"


def evaluate_command(
    args: argparse.Namespace, paths: RecognitionPaths, checkpoint: Path
) -> list[str]:
    options = common_options(args, paths)
    options.extend(
        [
            ("Global.pretrained_model", ""),
            ("Global.checkpoints", resolve_checkpoint_prefix(checkpoint)),
        ]
    )
    return options_command(
        Path(sys.executable),
        paths.paddleocr_dir / "tools" / "eval.py",
        paths.config,
        options,
    )


def export_command(
    args: argparse.Namespace,
    paths: RecognitionPaths,
    checkpoint: Path,
    staging_dir: Path,
) -> list[str]:
    options = common_options(args, paths)
    options.extend(
        [
            # Paddle 3.2 exports this SVTR model reliably through PIR. Keep
            # the training config portable, but use PIR for inference export.
            ("Global.export_with_pir", True),
            ("Global.pretrained_model", ""),
            ("Global.checkpoints", resolve_checkpoint_prefix(checkpoint)),
            ("Global.save_inference_dir", staging_dir),
        ]
    )
    return options_command(
        Path(sys.executable),
        paths.paddleocr_dir / "tools" / "export_model.py",
        paths.config,
        options,
    )


def parse_training_progress(
    line: str,
) -> tuple[int, int, int, str | None] | None:
    """Extract progress fields from PaddleOCR's training log line."""
    progress_match = TRAIN_PROGRESS_RE.search(line)
    if not progress_match:
        return None

    loss_match = LOSS_RE.search(line)
    return (
        int(progress_match.group("epoch")),
        int(progress_match.group("epochs")),
        int(progress_match.group("step")),
        loss_match.group("loss") if loss_match else None,
    )


def estimate_training_steps(
    train_samples: int, batch_size: int, epochs: int
) -> int:
    """Estimate tqdm's total from the dataset and PaddleOCR batch settings."""
    steps_per_epoch = max(1, train_samples // batch_size)
    return steps_per_epoch * epochs


def run_command(
    command: list[str],
    cwd: Path,
    *,
    show_progress: bool = False,
    total_steps: int | None = None,
) -> None:
    require_path(Path(command[1]), "PaddleOCR entrypoint")
    print(f"Running: {shlex.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    progress = (
        tqdm(
            total=total_steps,
            desc="Recognition train",
            unit="step",
            dynamic_ncols=True,
        )
        if show_progress
        else None
    )
    last_step = 0

    try:
        assert process.stdout is not None
        for line in process.stdout:
            parsed = parse_training_progress(line)
            if parsed and progress is not None:
                epoch, epochs, step, loss = parsed
                progress.update(max(0, step - last_step))
                last_step = max(last_step, step)
                progress.set_postfix(
                    epoch=f"{epoch}/{epochs}",
                    loss=loss or "n/a",
                    refresh=False,
                )
                continue

            if progress is not None:
                tqdm.write(line.rstrip())
            else:
                sys.stdout.write(line)
                sys.stdout.flush()

        return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()
        if progress is not None:
            progress.close()


def normalize_export(staging_dir: Path, inference_dir: Path) -> None:
    """Put PaddleOCR's exported files and config in one PaddleX model folder."""
    params_files = sorted(staging_dir.rglob("inference.pdiparams"))
    if not params_files:
        raise FileNotFoundError(
            f"PaddleOCR export completed without inference.pdiparams in {staging_dir}"
        )
    source_dir = params_files[0].parent
    config_files = sorted(
        staging_dir.rglob("inference.yml"),
        key=lambda path: len(path.relative_to(staging_dir).parts),
    )
    if not config_files:
        raise FileNotFoundError(
            f"PaddleOCR export completed without inference.yml in {staging_dir}"
        )

    inference_dir.mkdir(parents=True, exist_ok=True)
    for source in source_dir.glob("inference.*"):
        if source.is_file():
            shutil.copy2(source, inference_dir / source.name)
    config_source = source_dir / "inference.yml"
    if not config_source.is_file():
        config_source = config_files[0]
    shutil.copy2(config_source, inference_dir / "inference.yml")

    required = (inference_dir / "inference.pdiparams", inference_dir / "inference.yml")
    missing = [path.name for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Could not normalize exported recognition model; missing {missing}"
        )
    print(f"Exported recognition model: {inference_dir}")


def export_model(
    args: argparse.Namespace, paths: RecognitionPaths, checkpoint: Path
) -> None:
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        dir=paths.output_dir, prefix=".recognition-export-"
    ) as temporary_dir:
        staging_dir = Path(temporary_dir)
        run_command(
            export_command(args, paths, checkpoint, staging_dir),
            cwd=paths.paddleocr_dir,
        )
        normalize_export(staging_dir, paths.inference_dir)


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    train_samples = prepare_dataset(paths, args.max_text_length)

    if args.mode == "check_dataset":
        return

    require_path(paths.config, "recognition config")
    require_path(paths.paddleocr_dir, "PaddleOCR source directory")
    require_path(paths.paddleocr_dir / "tools", "PaddleOCR tools directory")

    if args.mode == "train":
        if args.pretrained_model and not args.pretrained_model.startswith(("http://", "https://")):
            pretrained_path = Path(args.pretrained_model).expanduser().resolve()
            require_path(pretrained_path, "pretrained model")
        run_command(
            train_command(args, paths),
            cwd=paths.paddleocr_dir,
            show_progress=args.progress,
            total_steps=estimate_training_steps(
                train_samples, args.batch_size, args.epochs
            ),
        )
        export_model(args, paths, default_checkpoint(paths))
        return

    checkpoint = args.weights or default_checkpoint(paths)
    if args.mode == "evaluate":
        run_command(
            evaluate_command(args, paths, checkpoint),
            cwd=paths.paddleocr_dir,
        )
        return

    export_model(args, paths, checkpoint)


if __name__ == "__main__":
    main()
