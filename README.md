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

The pretrained and inference checkpoints are stored with Git LFS. Install Git
LFS before cloning or pulling the repository, then run `git lfs pull` to fetch
the actual weight files.

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

# Review 10 layout predictions; green boxes in comparison/ are ground truth
CUDA_VISIBLE_DEVICES=MIG-b41c5bd0-f3d2-564b-859a-e6f2e060b5ee \
  ./OCR/bin/python detect_layout.py --count 10 --device gpu:0

# Run OCR text recognition on the same 10 bills
CUDA_VISIBLE_DEVICES=MIG-b41c5bd0-f3d2-564b-859a-e6f2e060b5ee \
  ./OCR/bin/python infer_sample.py --device gpu:0
```

### Fine-tune text recognition

The prepared recognition dataset is read by default from
`/home/tntri/data/OCR_dataset`: 6,585 crop images with
`text_recognition_train_data.txt` and `text_recognition_val_data.txt`. The
script checks every image/label, generates a UTF-8 character dictionary from
the labels, fine-tunes `PP-OCRv5_server_rec`, evaluates it, and exports the
model to `outputs/recognition/inference`.

```bash
# Check labels and generate the dataset-specific dictionary
./OCR/bin/python train_text_recognition.py --mode check_dataset

# Fine-tune on one MIG slice; training automatically exports the best model
./run_train_rec.sh --list-mig
./run_train_rec.sh --mig 0

# Optional overrides
./run_train_rec.sh --mig 0 --epochs 30 --batch-size 8 --learning-rate 0.0001

# Evaluate or re-export a checkpoint later
./OCR/bin/python train_text_recognition.py --mode evaluate
./OCR/bin/python train_text_recognition.py --mode export

# Evaluate 10 random validation crops directly with the recognition model
CUDA_VISIBLE_DEVICES=MIG-b41c5bd0-f3d2-564b-859a-e6f2e060b5ee \
  ./OCR/bin/python infer_recognition_sample.py --device gpu:0
```

The dictionary excludes the literal space because PaddleOCR adds it through
`use_space_char: true`. `--max-text-length` defaults to 160 because the
current labels are longer than PP-OCRv5's stock limit of 25. Since the custom
dictionary contains Vietnamese characters absent from the stock dictionary,
the recognition output heads are initialized for this project while the
backbone/encoder is loaded from the official pretrained checkpoint.

After export, `model.py` automatically detects
`outputs/recognition/inference`, so the existing `infer.py` and
`infer_sample.py` commands use the fine-tuned text encoder/recognizer without
another code change. To use a different model location, pass
`--recognition-output-dir` to those inference commands.

The layout review is saved in `outputs/layout_sample_10/`: `predictions/` has
model boxes, `comparison/` overlays ground truth in green, and the CSV files
contain per-image and per-label miss/extra counts.

The OCR sample is saved in `outputs/ocr_sample_10/`: `layout_text.csv` contains
text grouped by the four custom layout labels, `all_text.csv` contains all OCR
text and confidence scores, and `visualizations/` contains OCR box images.

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
