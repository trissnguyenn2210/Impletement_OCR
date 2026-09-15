"""Run the fine-tuned text recognizer on random labeled crops and visualize them."""

from __future__ import annotations

import argparse
import csv
import json
import random
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from tqdm import tqdm

from config import CONFIG
from model import _allow_headless_opencv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--labels",
        type=Path,
        default=CONFIG.recognition_val_labels,
        help="Label file containing '<image>\\t<text>' records",
    )
    parser.add_argument("--data-dir", type=Path, default=CONFIG.recognition_data_dir)
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=CONFIG.recognition_output_dir / "inference",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CONFIG.project_dir / "outputs" / "recognition_sample_10",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260915)
    parser.add_argument("--device", default=CONFIG.device)
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be >= 1")
    return args


def read_records(label_path: Path, data_dir: Path) -> list[tuple[Path, str]]:
    records = []
    for line_number, line in enumerate(
        label_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line:
            continue
        if "\t" not in line:
            raise ValueError(f"{label_path}:{line_number} is missing a tab")
        image_name, ground_truth = line.split("\t", 1)
        image_path = Path(image_name)
        if not image_path.is_absolute():
            image_path = data_dir / image_path
        if not image_path.is_file():
            raise FileNotFoundError(f"Image not found: {image_path}")
        records.append((image_path, ground_truth))
    if not records:
        raise ValueError(f"No records found in {label_path}")
    return records


def normalize_text(value: str) -> str:
    return " ".join(value.split())


def get_font(size: int) -> ImageFont.ImageFont:
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if font_path.is_file():
        return ImageFont.truetype(str(font_path), size)
    return ImageFont.load_default()


def draw_card(
    image_path: Path,
    ground_truth: str,
    prediction: str,
    score: float,
    *,
    width: int = 1000,
    height: int = 300,
) -> Image.Image:
    exact_match = normalize_text(ground_truth) == normalize_text(prediction)
    background = (232, 255, 232) if exact_match else (255, 235, 235)
    card = Image.new("RGB", (width, height), background)
    draw = ImageDraw.Draw(card)

    title_font = get_font(24)
    body_font = get_font(21)
    status = "OK" if exact_match else "MISMATCH"
    status_color = (0, 125, 45) if exact_match else (185, 25, 25)
    draw.text((18, 12), f"{status} | {image_path.name}", fill=status_color, font=title_font)

    caption = (
        f"GT: {ground_truth}\n"
        f"Pred: {prediction}\n"
        f"Score: {score:.4f}"
    )
    draw.multiline_text(
        (18, 48),
        textwrap.fill(caption, width=92, replace_whitespace=False),
        fill=(20, 20, 20),
        font=body_font,
        spacing=4,
    )

    image = Image.open(image_path).convert("RGB")
    image = ImageOps.contain(image, (width - 36, height - 150))
    x = (width - image.width) // 2
    y = height - image.height - 12
    card.paste(image, (x, y))
    return card


def main() -> None:
    args = parse_args()
    labels_path = args.labels.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Inference model not found: {model_dir}. Export the recognition model first."
        )

    records = read_records(labels_path, data_dir)
    if args.count > len(records):
        raise ValueError(f"Requested {args.count} samples, only {len(records)} available")
    selected = random.Random(args.seed).sample(records, args.count)

    # PaddleX checks for the GUI OpenCV package even for image-only inference;
    # the project intentionally installs the headless package on the server.
    _allow_headless_opencv()
    from paddleocr import TextRecognition

    recognizer = TextRecognition(model_dir=str(model_dir), device=args.device)
    rows = []
    cards = []
    try:
        for index, (image_path, ground_truth) in enumerate(
            tqdm(selected, desc="Recognition inference", unit="image"), start=1
        ):
            result = recognizer.predict(input=str(image_path), batch_size=1)[0]
            prediction = str(result["rec_text"])
            score = float(result["rec_score"])
            exact_match = ground_truth == prediction
            normalized_match = normalize_text(ground_truth) == normalize_text(prediction)
            row = {
                "image": str(image_path),
                "ground_truth": ground_truth,
                "prediction": prediction,
                "score": score,
                "exact_match": exact_match,
                "normalized_match": normalized_match,
            }
            rows.append(row)
            card = draw_card(image_path, ground_truth, prediction, score)
            card.save(output_dir / f"sample_{index:02d}_{image_path.stem}.jpg", quality=95)
            cards.append(card)
            print(
                f"[{index:02d}] {'OK' if normalized_match else 'MISS'} "
                f"score={score:.4f} | GT={ground_truth!r} | Pred={prediction!r}"
            )
    finally:
        recognizer.close()

    columns = 2
    rows_count = (len(cards) + columns - 1) // columns
    contact_sheet = Image.new("RGB", (columns * 1000, rows_count * 300), (210, 210, 210))
    for index, card in enumerate(cards):
        contact_sheet.paste(card, ((index % columns) * 1000, (index // columns) * 300))
    contact_sheet.save(output_dir / "contact_sheet.jpg", quality=95)

    with (output_dir / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "results.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    normalized_accuracy = sum(row["normalized_match"] for row in rows) / len(rows)
    print(f"Normalized exact-match: {normalized_accuracy:.1%} ({sum(row['normalized_match'] for row in rows)}/{len(rows)})")
    print(f"Contact sheet: {output_dir / 'contact_sheet.jpg'}")
    print(f"Results CSV: {output_dir / 'results.csv'}")


if __name__ == "__main__":
    main()
