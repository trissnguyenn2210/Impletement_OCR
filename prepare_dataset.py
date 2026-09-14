"""Convert the MCOCR CSV annotations to PaddleX COCO detection format."""

import argparse
import ast
import csv
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image


LABEL_ORDER = ["SELLER", "ADDRESS", "TIMESTAMP", "TOTAL_COST"]
LABEL_ALIASES = {"TOTAL_TOTAL_COST": "TOTAL_COST"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="/home/tntri/data/OCR_dataset/mcocr_train_df.csv",
        help="MCOCR CSV file",
    )
    parser.add_argument(
        "--images",
        default="/home/tntri/data/OCR_dataset/train_images/train_images",
        help="Directory containing the CSV images",
    )
    parser.add_argument(
        "--output",
        default="/home/tntri/data/OCR_dataset_coco",
        help="Output COCO dataset directory",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalize_label(label: str) -> str:
    label = LABEL_ALIASES.get(label.strip(), label.strip())
    if label not in LABEL_ORDER:
        raise ValueError(f"Unexpected label: {label}")
    return label


def image_size(image_path: Path, polygons: list[dict]) -> tuple[int, int]:
    if polygons:
        width = int(polygons[0]["width"])
        height = int(polygons[0]["height"])
        if any(
            int(item["width"]) != width or int(item["height"]) != height
            for item in polygons
        ):
            raise ValueError(f"Inconsistent image size in annotations: {image_path.name}")
        return width, height

    with Image.open(image_path) as image:
        return image.size


def read_records(csv_path: Path, images_dir: Path) -> tuple[list[dict], Counter]:
    records = []
    label_counts = Counter()

    with csv_path.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            image_name = row["img_id"]
            image_path = images_dir / image_name
            if not image_path.is_file():
                raise FileNotFoundError(f"Image not found: {image_path}")

            polygons = ast.literal_eval(row["anno_polygons"] or "[]")
            labels = [normalize_label(label) for label in row["anno_labels"].split("|||") if label.strip()]
            if len(polygons) != len(labels):
                raise ValueError(
                    f"Polygon/label count mismatch for {image_name}: "
                    f"{len(polygons)} != {len(labels)}"
                )

            width, height = image_size(image_path, polygons)
            annotations = []
            for polygon, label in zip(polygons, labels):
                x, y, box_width, box_height = map(float, polygon["bbox"])
                if box_width <= 0 or box_height <= 0:
                    raise ValueError(f"Invalid bbox in {image_name}: {polygon['bbox']}")

                category_id = LABEL_ORDER.index(label) + 1
                annotations.append(
                    {
                        "category_id": category_id,
                        "bbox": [x, y, box_width, box_height],
                        "area": float(polygon.get("area", box_width * box_height)),
                        "iscrowd": 0,
                        "segmentation": polygon.get("segmentation", []),
                    }
                )
                label_counts[label] += 1

            records.append(
                {
                    "file_name": image_name,
                    "width": width,
                    "height": height,
                    "annotations": annotations,
                    "source": image_path,
                }
            )

    return records, label_counts


def make_coco(records: list[dict]) -> dict:
    images = []
    annotations = []
    annotation_id = 1

    for image_id, record in enumerate(records, start=1):
        images.append(
            {
                "id": image_id,
                "file_name": record["file_name"],
                "width": record["width"],
                "height": record["height"],
            }
        )
        for item in record["annotations"]:
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": image_id,
                    **item,
                }
            )
            annotation_id += 1

    categories = [
        {"id": index, "name": label, "supercategory": "bill_field"}
        for index, label in enumerate(LABEL_ORDER, start=1)
    ]
    return {"images": images, "annotations": annotations, "categories": categories}


def main() -> None:
    args = parse_args()
    if not 0 < args.val_ratio < 1:
        raise ValueError("--val-ratio must be between 0 and 1")

    csv_path = Path(args.csv).expanduser().resolve()
    images_dir = Path(args.images).expanduser().resolve()
    output_dir = Path(args.output).expanduser().resolve()
    records, label_counts = read_records(csv_path, images_dir)

    random.Random(args.seed).shuffle(records)
    val_size = max(1, int(len(records) * args.val_ratio))
    val_records = records[:val_size]
    train_records = records[val_size:]

    output_images = output_dir / "images"
    output_annotations = output_dir / "annotations"
    output_images.mkdir(parents=True, exist_ok=True)
    output_annotations.mkdir(parents=True, exist_ok=True)

    for record in records:
        shutil.copy2(record["source"], output_images / record["file_name"])

    for name, split in (("instance_train.json", train_records), ("instance_val.json", val_records)):
        target = output_annotations / name
        target.write_text(json.dumps(make_coco(split), ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Output: {output_dir}")
    print(f"Images: {len(records)} ({len(train_records)} train / {len(val_records)} val)")
    print(f"Labels: {dict(label_counts)}")


if __name__ == "__main__":
    main()
