# Dataset

`train.py` expects a PaddleX `COCODetDataset` directory. Keep the dataset outside
the repository if it is large and pass its absolute path with `--dataset-dir`.

The annotation files must contain the same category order as
`configs/layout_labels.txt`. Update that file and the COCO categories together.

Expected layout:

```text
dataset/
├── annotations/
│   ├── instance_train.json
│   └── instance_val.json
└── images/
```

Both JSON files use the COCO detection format. The `file_name` values should
point to files under `images/`.

For the MCOCR dataset currently available on this machine, convert the CSV
without changing the original data:

```bash
./OCR/bin/python prepare_dataset.py
```

This creates `/home/tntri/data/OCR_dataset_coco` and makes a deterministic 80/20
split from the images annotated in `mcocr_train_df.csv`. The original validation
images are not used for layout validation because their CSV has no polygons or
layout labels.

The exact format can be checked before training:

```bash
./OCR/bin/python train.py --mode check_dataset \
  --dataset-dir /home/tntri/data/OCR_dataset_coco
```

The same command can be run through uv:

```bash
./run_uv.sh --mode check_dataset \
  --dataset-dir /home/tntri/data/OCR_dataset_coco
```
