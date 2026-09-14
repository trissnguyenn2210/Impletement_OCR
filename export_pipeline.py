"""Export a PP-StructureV3 YAML using the current fine-tuned model path."""

import argparse
from dataclasses import replace
from pathlib import Path

from config import CONFIG
from model import build_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=CONFIG.pipeline_config)
    parser.add_argument(
        "--layout-output-dir",
        type=Path,
        default=CONFIG.output_dir,
        help="Directory containing the fine-tuned layout model",
    )
    parser.add_argument("--model-name", default=CONFIG.model_name)
    parser.add_argument("--device", default=CONFIG.device)
    args = parser.parse_args()

    config = replace(
        CONFIG,
        output_dir=args.layout_output_dir.expanduser().resolve(),
        model_name=args.model_name,
        device=args.device,
    )
    if not config.finetuned_layout_dir.exists():
        raise FileNotFoundError(
            f"Train the layout model first: {config.finetuned_layout_dir}"
        )

    pipeline = build_pipeline(config)
    pipeline.export_paddlex_config_to_yaml(str(args.output))
    print(f"Saved pipeline config to {args.output}")


if __name__ == "__main__":
    main()
