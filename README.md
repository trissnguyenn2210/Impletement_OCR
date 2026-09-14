# OCR_project

Starter project for Vietnamese/English bill OCR with PaddleOCR + PP-StructureV3.
The project fine-tunes the layout detector first, then feeds the trained layout
model into the full PP-StructureV3 pipeline for OCR and table parsing.

## Why this design

PP-StructureV3 is a pipeline, not one single checkpoint. Its layout detection,
text detection, text recognition, and table recognition modules can be tuned
independently. For bills, the first useful adaptation is usually the layout
detector; text recognition should be fine-tuned later if the bill font, language,
or image quality causes OCR errors.

The downloaded starting checkpoint is:

```text
models/PP-DocLayout_plus-L_pretrained.pdparams
```

## Install

This machine has an NVIDIA A100 and driver support for the CUDA 12.6 wheel.
Run:

```bash
cd /home/tntri/OCR_project
./setup_env.sh
```

For another GPU/CPU, change the PaddlePaddle wheel index in `setup_env.sh`.
The project also includes `pyproject.toml` for uv. After setup, Python commands
can use `./OCR/bin/python` directly. The `run_uv.sh` wrapper
uses the same `OCR` environment without creating a second `.venv`.

## Dataset

Put or point to a PaddleX `COCODetDataset` directory. See `data/README.md`.
Before training, replace the example categories in
`configs/layout_labels.txt` with the exact categories in your annotations.

For the MCOCR CSV dataset at `/home/tntri/data/OCR_dataset`, prepare the COCO
dataset first:

```bash
./OCR/bin/python prepare_dataset.py
```

## Commands

```bash
# Validate annotations
./OCR/bin/python train.py --mode check_dataset \
  --dataset-dir /home/tntri/data/OCR_dataset_coco

# Fine-tune PP-DocLayout_plus-L
./run_uv.sh --mode train \
  --dataset-dir /home/tntri/data/OCR_dataset_coco

# Fine-tune on a selected MIG slice; batch size and workers come from config.py
./run_train.sh --list-mig
./run_train.sh --mig 0 --epochs 100

# Early stopping mặc định: dừng sau 20 lần validation liên tiếp không tăng bbox mAP
./run_train.sh --mig 0 --early-stopping-patience 20

# Tắt early stopping nếu cần
./run_train.sh --mig 0 --early-stopping-patience 0

# Or run without MIG through uv
./run_uv.sh --mode train

# Override the number of data-loader workers
./run_train.sh --mig 0 --num-workers 4
./run_uv.sh --mode train --num-workers 4

# Evaluate the best checkpoint
./OCR/bin/python train.py --mode evaluate \
  --dataset-dir /home/tntri/data/OCR_dataset_coco

# Run the complete PP-StructureV3 pipeline
./OCR/bin/python infer.py /absolute/path/to/bill.jpg
```

Training logs are saved in `outputs/layout/`. Mỗi lần chạy mới sẽ ghi đè các
file theo dõi của lần chạy trước:

- `train.log`: raw PaddleDetection text log
- `train_metrics.csv`: iteration metrics for analysis
- `train_summary.txt`: short human-readable summary
- `train_result.json`: checkpoint and evaluation results

VisualDL được tắt mặc định để không tạo thêm các file `vdlrecords.*.log`; tqdm
được dùng để hiển thị tiến trình trực tiếp trên terminal.

The CSV file is updated trong lúc training và được ghi lại sau khi kết thúc.
To convert a currently existing log manually:

```bash
./run_uv.sh --mode check_dataset  # verify the environment
./OCR/bin/python summarize_log.py
```

Useful overrides:

```bash
./OCR/bin/python train.py --mode train \
  --dataset-dir /home/tntri/data/OCR_dataset_coco \
  --device gpu:0 --batch-size 4 --epochs 80

# Enable correction for photographed/rotated bills when needed
./OCR/bin/python infer.py bill.jpg \
  --doc-orientation --doc-unwarping
```

The output layout model is expected at
`outputs/layout/best_model/inference`. `model.py` automatically uses it when
present; otherwise PP-StructureV3 falls back to the official model by name.

## Repository contents

The repository keeps source code, configs, the pretrained checkpoint, the best
inference model, and the latest training summaries. The virtual environment,
PaddleX source checkout, dataset, intermediate checkpoints, and system caches
are intentionally kept local. Install dependencies with `setup_env.sh`; it also
reapplies the small PaddleX/PaddleDetection patches stored in `patches/`.

## Next step after the first dataset

The current layout labels are `SELLER`, `ADDRESS`, `TIMESTAMP`, and `TOTAL_COST`.
Keep a held-out set from different vendors/templates. After measuring errors,
fine-tune only the module responsible for the failure instead of retraining the
entire pipeline.
