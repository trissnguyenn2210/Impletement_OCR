"""Validate, train, evaluate, or predict the layout model through PaddleX."""

import argparse
import csv
import subprocess
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from config import CONFIG, Config
from summarize_log import CSV_FIELDS, parse_log, parse_training_line, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        nargs="?",
        choices=("check_dataset", "train", "evaluate", "predict"),
        help="Operation to run (can also be passed as --mode)",
    )
    parser.add_argument(
        "--mode",
        dest="mode_option",
        choices=("check_dataset", "train", "evaluate", "predict"),
        help="Operation to run",
    )
    parser.add_argument("--input", type=Path, help="Image/PDF path for predict mode")
    parser.add_argument("--weights", type=Path, help="Weights path for evaluate mode")
    parser.add_argument("--dataset-dir", type=Path, default=CONFIG.dataset_dir)
    parser.add_argument("--output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument("--paddlex-dir", type=Path, default=CONFIG.paddlex_dir)
    parser.add_argument("--pretrain-path", type=Path, default=CONFIG.pretrain_path)
    parser.add_argument("--layout-config", type=Path, default=CONFIG.layout_config)
    parser.add_argument("--labels-file", type=Path, default=CONFIG.labels_file)
    parser.add_argument("--model-name", default=CONFIG.model_name)
    parser.add_argument("--device", default=CONFIG.device)
    parser.add_argument("--epochs", type=int, default=CONFIG.epochs)
    parser.add_argument("--batch-size", type=int, default=CONFIG.batch_size)
    parser.add_argument("--num-workers", type=int, default=CONFIG.num_workers)
    parser.add_argument("--learning-rate", type=float, default=CONFIG.learning_rate)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=CONFIG.early_stopping_patience,
        help="Stop after this many validations without higher bbox mAP; 0 disables it",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=CONFIG.early_stopping_min_delta,
        help="Minimum bbox mAP increase required to reset early-stopping patience",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a tqdm progress bar during training",
    )
    args = parser.parse_args()
    if args.mode and args.mode_option and args.mode != args.mode_option:
        parser.error("positional mode and --mode must match")
    if args.early_stopping_patience < 0:
        parser.error("--early-stopping-patience must be >= 0")
    if args.early_stopping_min_delta < 0:
        parser.error("--early-stopping-min-delta must be >= 0")
    args.mode = args.mode_option or args.mode
    args.mode = args.mode or "train"
    return args


def require_path(path: Path, name: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{name} not found: {path}")


def make_config(args: argparse.Namespace) -> Config:
    path_fields = {
        name: getattr(args, name).expanduser().resolve()
        for name in (
            "dataset_dir",
            "output_dir",
            "paddlex_dir",
            "pretrain_path",
            "layout_config",
            "labels_file",
        )
    }
    return replace(
        CONFIG,
        **path_fields,
        model_name=args.model_name,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        learning_rate=args.learning_rate,
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_min_delta=args.early_stopping_min_delta,
    )


def make_command(config: Config, args: argparse.Namespace) -> list[str]:
    require_path(config.paddlex_main, "PaddleX main.py")
    command = [
        sys.executable,
        str(config.paddlex_main),
        "-c",
        str(config.layout_config),
        "-o",
        f"Global.mode={args.mode}",
        "-o",
        f"Global.model={config.model_name}",
        "-o",
        f"Global.dataset_dir={config.dataset_dir}",
        "-o",
        f"Global.device={config.device}",
        "-o",
        f"Global.output={config.output_dir}",
        "-o",
        f"worker_num={config.num_workers}",
    ]

    if args.mode == "train":
        require_path(config.pretrain_path, "pretrained weights")
        command.extend(
            [
                "-o",
                f"Train.num_classes={config.num_classes}",
                "-o",
                f"Train.epochs_iters={config.epochs}",
                "-o",
                f"Train.batch_size={config.batch_size}",
                "-o",
                f"Train.learning_rate={config.learning_rate}",
                "-o",
                f"Train.pretrain_weight_path={config.pretrain_path}",
                "-o",
                f"early_stopping_patience={config.early_stopping_patience}",
                "-o",
                f"early_stopping_min_delta={config.early_stopping_min_delta}",
            ]
        )

    if args.mode == "evaluate":
        weight_path = args.weights or config.output_dir / "best_model" / "best_model.pdparams"
        command.extend(["-o", f"Evaluate.weight_path={weight_path.resolve()}"])

    if args.mode == "predict":
        if not args.input:
            raise ValueError("--input is required in predict mode")
        model_dir = config.finetuned_layout_dir
        require_path(model_dir, "fine-tuned inference model")
        command.extend(
            [
                "-o",
                f"Predict.model_dir={model_dir}",
                "-o",
                f"Predict.input={args.input.expanduser().resolve()}",
            ]
        )

    if args.mode in {"check_dataset", "train", "evaluate"}:
        require_path(config.dataset_dir, "dataset directory")

    return command


def follow_training_log(
    log_path: Path,
    csv_path: Path,
    epochs: int,
    stop_event: threading.Event,
    show_progress: bool,
) -> None:
    """Record training metrics to CSV and optionally display them with tqdm."""
    # train.log is reset before this thread starts, so always read this run
    # from the beginning.
    position = 0
    progress = None
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            writer.writeheader()

            while True:
                if not log_path.exists():
                    if stop_event.is_set():
                        break
                    time.sleep(0.2)
                    continue

                size = log_path.stat().st_size
                if size < position:
                    position = 0

                with log_path.open(encoding="utf-8") as log_file:
                    log_file.seek(position)
                    while True:
                        line = log_file.readline()
                        if not line:
                            break
                        position = log_file.tell()
                        row = parse_training_line(line)
                        if not row:
                            continue

                        writer.writerow(row)
                        csv_file.flush()
                        if not show_progress:
                            continue

                        epoch = int(row["epoch"])
                        step = int(row["iteration"])
                        steps = int(row["iterations_per_epoch"])
                        if progress is None:
                            progress = tqdm(
                                total=max(1, epochs) * steps,
                                desc="Layout train",
                                unit="iter",
                                dynamic_ncols=True,
                            )

                        current = epoch * steps + step
                        progress.update(max(0, current - progress.n))
                        if row["loss"]:
                            progress.set_postfix(loss=row["loss"], refresh=False)

                if stop_event.is_set():
                    break
                time.sleep(0.2)
    finally:
        if progress is not None:
            progress.close()


def run_command(
    config: Config,
    command: list[str],
    track_training: bool,
    show_progress: bool,
    epochs: int,
) -> None:
    if not track_training:
        subprocess.run(command, cwd=config.paddlex_dir, check=True)
        return

    stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=follow_training_log,
        args=(
            config.output_dir / "train.log",
            config.output_dir / "train_metrics.csv",
            epochs,
            stop_event,
            show_progress,
        ),
        daemon=True,
    )
    progress_thread.start()
    try:
        subprocess.run(command, cwd=config.paddlex_dir, check=True)
    finally:
        stop_event.set()
        progress_thread.join()


def reset_training_logs(config: Config) -> None:
    """Start each training run with one fresh raw log and one fresh CSV."""
    for filename in ("train.log", "train_metrics.csv"):
        (config.output_dir / filename).write_text("", encoding="utf-8")
    for path in config.output_dir.glob("vdlrecords.*.log"):
        path.unlink()


def export_training_csv(config: Config) -> None:
    try:
        rows, _ = parse_log(config.output_dir / "train.log")
        write_csv(rows, config.output_dir / "train_metrics.csv")
        print(f"CSV metrics: {config.output_dir / 'train_metrics.csv'}")
    except FileNotFoundError:
        print("Training log was not created; CSV was skipped.", file=sys.stderr)


def main() -> None:
    args = parse_args()
    config = make_config(args)
    command = make_command(config, args)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "train":
            reset_training_logs(config)
        run_command(config, command, args.mode == "train", args.progress, args.epochs)
    finally:
        if args.mode == "train":
            export_training_csv(config)


if __name__ == "__main__":
    main()
