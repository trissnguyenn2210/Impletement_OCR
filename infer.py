"""Run PP-StructureV3 and save structured JSON/Markdown results."""

import argparse
from dataclasses import replace
from pathlib import Path

from config import CONFIG
from model import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Bill image, PDF, or directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=CONFIG.project_dir / "outputs" / "inference",
    )
    parser.add_argument(
        "--layout-output-dir",
        type=Path,
        default=CONFIG.output_dir,
        help="Directory containing the fine-tuned layout model",
    )
    parser.add_argument(
        "--recognition-output-dir",
        type=Path,
        default=CONFIG.recognition_output_dir,
        help="Directory containing the fine-tuned text-recognition model",
    )
    parser.add_argument("--model-name", default=CONFIG.model_name)
    parser.add_argument("--device", default=CONFIG.device)
    parser.add_argument(
        "--doc-orientation",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.use_doc_orientation,
    )
    parser.add_argument(
        "--doc-unwarping",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.use_doc_unwarping,
    )
    parser.add_argument(
        "--textline-orientation",
        action=argparse.BooleanOptionalAction,
        default=CONFIG.use_textline_orientation,
    )
    args = parser.parse_args()

    config = replace(
        CONFIG,
        output_dir=args.layout_output_dir.expanduser().resolve(),
        recognition_output_dir=args.recognition_output_dir.expanduser().resolve(),
        model_name=args.model_name,
        device=args.device,
        use_doc_orientation=args.doc_orientation,
        use_doc_unwarping=args.doc_unwarping,
        use_textline_orientation=args.textline_orientation,
    )
    output_dir = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = build_pipeline(config)
    for result in pipeline.predict(input=str(args.input.expanduser().resolve())):
        result.save_to_json(save_path=str(output_dir))
        result.save_to_markdown(save_path=str(output_dir))
        result.print()


if __name__ == "__main__":
    main()
