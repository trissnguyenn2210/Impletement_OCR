"""Convert PaddleDetection training logs to CSV and a readable summary."""

import argparse
import csv
import json
import re
from pathlib import Path

from config import CONFIG


PROGRESS_RE = re.compile(
    r"\[(?P<timestamp>\d{2}/\d{2} \d{2}:\d{2}:\d{2})\].*?"
    r"Epoch:\s*\[(?P<epoch>\d+)\]\s*"
    r"\[\s*(?P<iteration>\d+)\s*/\s*(?P<total>\d+)\]"
)
BEST_AP_RE = re.compile(r"Best test bbox ap is\s+(?P<score>[-+]?\d+(?:\.\d+)?)")

METRIC_PATTERNS = {
    "learning_rate": r"learning_rate:\s*(\S+)",
    "loss": r"\bloss:\s*(\S+)",
    "batch_cost_sec": r"batch_cost:\s*(\S+)",
    "data_cost_sec": r"data_cost:\s*(\S+)",
    "images_per_sec": r"ips:\s*(\S+)",
    "max_mem_reserved_mb": r"max_mem_reserved:\s*(\S+)",
    "max_mem_allocated_mb": r"max_mem_allocated:\s*(\S+)",
}

CSV_FIELDS = [
    "timestamp",
    "epoch",
    "iteration",
    "iterations_per_epoch",
    "learning_rate",
    "loss",
    "batch_cost_sec",
    "data_cost_sec",
    "images_per_sec",
    "max_mem_reserved_mb",
    "max_mem_allocated_mb",
]


def parse_value(pattern: str, line: str) -> str:
    match = re.search(pattern, line)
    return match.group(1) if match else ""


def parse_training_line(line: str) -> dict[str, str] | None:
    progress = PROGRESS_RE.search(line)
    if not progress:
        return None

    row = {
        "timestamp": progress.group("timestamp"),
        "epoch": progress.group("epoch"),
        "iteration": str(int(progress.group("iteration")) + 1),
        "iterations_per_epoch": progress.group("total"),
    }
    row.update(
        {
            name: parse_value(pattern, line)
            for name, pattern in METRIC_PATTERNS.items()
        }
    )
    return row


def parse_log(log_path: Path) -> tuple[list[dict[str, str]], list[float]]:
    rows = []
    best_scores = []

    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        best_match = BEST_AP_RE.search(line)
        if best_match:
            best_scores.append(float(best_match.group("score")))

        row = parse_training_line(line)
        if row:
            rows.append(row)

    return rows, best_scores


def write_csv(rows: list[dict[str, str]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_training_result(output_dir: Path) -> tuple[str, str]:
    result_path = output_dir / "train_result.json"
    if not result_path.is_file():
        return "", ""

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "", ""

    best = result.get("models", {}).get("best", {})
    score = best.get("score", "")
    done = result.get("done_flag", "")
    return str(score), str(done)


def write_summary(
    log_path: Path,
    txt_path: Path,
    rows: list[dict[str, str]],
    best_scores: list[float],
) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "PaddleDetection training summary",
        f"Log: {log_path}",
        f"Metric records: {len(rows)}",
    ]

    if rows:
        last = rows[-1]
        losses = [float(row["loss"]) for row in rows if row["loss"]]
        lines.extend(
            [
                f"Last progress: epoch {last['epoch']}, "
                f"iteration {last['iteration']}/{last['iterations_per_epoch']}",
                f"Last loss: {last['loss']}",
                f"Minimum logged loss: {min(losses):.6f}" if losses else "",
                f"Last learning rate: {last['learning_rate']}",
                f"Last speed: {last['images_per_sec']} images/s",
                f"Last GPU reserved: {last['max_mem_reserved_mb']} MB",
            ]
        )

    result_score, done_flag = read_training_result(log_path.parent)
    scores = best_scores or ([float(result_score)] if result_score else [])
    if scores:
        lines.append(f"Best bbox AP: {max(scores):.6f}")
    if done_flag:
        lines.append(f"PaddleX done_flag: {done_flag}")

    txt_path.write_text("\n".join(line for line in lines if line) + "\n", encoding="utf-8")


def summarize_log(log_path: Path, csv_path: Path, txt_path: Path) -> None:
    if not log_path.is_file():
        raise FileNotFoundError(f"Training log not found: {log_path}")
    rows, best_scores = parse_log(log_path)
    write_csv(rows, csv_path)
    write_summary(log_path, txt_path, rows, best_scores)
    print(f"CSV summary: {csv_path}")
    print(f"TXT summary: {txt_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=CONFIG.output_dir / "train.log")
    parser.add_argument("--csv", type=Path, default=CONFIG.output_dir / "train_metrics.csv")
    parser.add_argument("--txt", type=Path, default=CONFIG.output_dir / "train_summary.txt")
    args = parser.parse_args()
    summarize_log(
        args.log.expanduser().resolve(),
        args.csv.expanduser().resolve(),
        args.txt.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()
