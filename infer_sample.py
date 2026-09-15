"""Run PP-StructureV3 OCR on a selected bill sample and save readable outputs."""

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path

from tqdm import tqdm

from config import CONFIG
from model import build_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-list",
        type=Path,
        default=CONFIG.project_dir / "outputs" / "layout_sample_10" / "selected_images.json",
        help="JSON list of image names. It is created by detect_layout.py.",
    )
    parser.add_argument("--input-dir", type=Path, default=CONFIG.dataset_dir / "images")
    parser.add_argument(
        "--output",
        type=Path,
        default=CONFIG.project_dir / "outputs" / "ocr_sample_10",
    )
    parser.add_argument("--layout-output-dir", type=Path, default=CONFIG.output_dir)
    parser.add_argument(
        "--recognition-output-dir",
        type=Path,
        default=CONFIG.recognition_output_dir,
    )
    parser.add_argument("--device", default=CONFIG.device)
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    names = json.loads(args.input_list.read_text(encoding="utf-8"))
    image_paths = [args.input_dir / name for name in names]
    missing = [path for path in image_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input image: {missing[0]}")

    output_dir = args.output.expanduser().resolve()
    json_dir = output_dir / "json"
    visualization_dir = output_dir / "visualizations"
    json_dir.mkdir(parents=True, exist_ok=True)
    visualization_dir.mkdir(parents=True, exist_ok=True)

    config = replace(
        CONFIG,
        output_dir=args.layout_output_dir.expanduser().resolve(),
        recognition_output_dir=args.recognition_output_dir.expanduser().resolve(),
        device=args.device,
    )
    pipeline = build_pipeline(
        config,
        use_table_recognition=False,
        use_region_detection=False,
    )

    all_results = []
    block_rows = []
    text_rows = []
    for image_path in tqdm(image_paths, desc="Run OCR", unit="bill"):
        result = pipeline.predict(input=str(image_path.resolve()))[0]
        data = result.json["res"]
        all_results.append(data)
        result.save_to_json(save_path=str(json_dir))
        result.save_to_img(save_path=str(visualization_dir))

        for block in data.get("parsing_res_list", []):
            block_rows.append(
                {
                    "image": image_path.name,
                    "label": block.get("block_label", ""),
                    "text": block.get("block_content", ""),
                    "bbox": block.get("block_bbox", []),
                }
            )

        ocr = data.get("overall_ocr_res", {})
        for bbox, text, score in zip(
            ocr.get("rec_boxes", []),
            ocr.get("rec_texts", []),
            ocr.get("rec_scores", []),
        ):
            text_rows.append(
                {
                    "image": image_path.name,
                    "text": text,
                    "score": score,
                    "bbox": bbox,
                }
            )

    (output_dir / "all_results.json").write_text(
        json.dumps(all_results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_csv(
        output_dir / "layout_text.csv",
        block_rows,
        ["image", "label", "text", "bbox"],
    )
    write_csv(
        output_dir / "all_text.csv",
        text_rows,
        ["image", "text", "score", "bbox"],
    )
    print(f"Saved OCR results to: {output_dir}")
    print(f"Layout-field text: {output_dir / 'layout_text.csv'}")
    print(f"All OCR text: {output_dir / 'all_text.csv'}")
    print(f"OCR visualizations: {visualization_dir}")


if __name__ == "__main__":
    main()
