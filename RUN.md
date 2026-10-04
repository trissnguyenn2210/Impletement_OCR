# RUN.md — Hướng dẫn chạy thực nghiệm

Dự án: **Visual Information Extraction for Visually Rich Vietnamese Receipts** (DSP391, Nhóm 3).

Mục tiêu của code trong `src/`: chạy 4 phương pháp trên 2 bộ dữ liệu, đo metric,
rồi **tự điền** vào 3 bảng trong `docs/experiment.xlsx`.

---

## 0. Trạng thái: cái gì đã kiểm chứng, cái gì chưa

Phải nói rõ ngay, vì nó quyết định bạn cần cẩn thận ở đâu:

| Phần | Đã chạy thật? |
|---|---|
| `build_gt.py`, `schema.py`, `metrics/*`, `methods/postproc.py`, `methods/dummy.py`, `run_eval.py`, `fill_excel.py`, `self_check.py`, `test_all.py` | **Có.** Chạy trên máy local với cả 2 dataset thật. 58/58 test pass. `fill_excel.py` đã điền thử 27 dòng và tôi đã kiểm tra từng ô. |
| `methods/real.py` (4 phương pháp thật) | **Chưa.** Máy local **không có `paddleocr` và không có GPU**. Tôi chỉ xác nhận được: file import được mà không cần paddle, 9 method đăng ký đúng tên, và hậu xử lý output (`postproc`) khớp đúng rule của GT. **Lời gọi API PaddleOCR/transformers bên trong chưa từng thực thi.** |

→ Vì vậy **bước 2 (`self_check.py`) là bắt buộc**, không phải tùy chọn. Nó chạy 3 ảnh
thay vì 1.153 ảnh, để lỗi API lộ ra trong 2 phút chứ không phải sau 3 giờ GPU.

---

## 1. Cài đặt

```bash
cd src
pip install -r ../requirements.txt     # nếu có
pip install openpyxl pandas pillow opencv-python rapidfuzz
# cho method A / C / D:
pip install paddlepaddle-gpu paddleocr
# cho method B / C / D:
pip install torch transformers accelerate
```

`rapidfuzz` là **tùy chọn** — không có thì `metrics/core.py` tự dùng Levenshtein
thuần Python, kết quả giống hệt, chỉ chậm hơn.

Đường dẫn ghi đè bằng biến môi trường, **không sửa code**:

```bash
export VIE_ROOT=/data/Impletement_OCR      # gốc project
export VIE_DATA=/data/datasets             # chứa MC-OCR/ và vietnamese-receipts-v3/
export VIE_WORK=/data/work                 # nơi ghi kết quả
export VIE_DEVICE=cuda:0
export VIE_VLM=Qwen/Qwen3-VL-4B-Instruct   # VLM cho B và C
export VIE_LLM=Qwen/Qwen3-4B-Instruct      # LLM cho D
export VIE_LAYOUT_DIR=/path/den/inference  # layout đã fine-tune (nếu không ở chỗ mặc định)
export VIE_REC_DIR=/path/den/inference     # recognition đã fine-tune
```

---

## 1.5. Tổng số lần chạy

| # | File | Số lần | Cần GPU? | Ghi ra gì |
|---|---|---|---|---|
| 1 | `train.py` (gốc project) | **2** | Có | checkpoint layout detector: 1 lần trên D1, 1 lần trên D2 |
| 2 | `train_text_recognition.py` (gốc project) | **2** | Có | checkpoint recognition: 1 lần trên D1, 1 lần trên D2 |
| 3 | `src/build_gt.py` | **1** | Không | 6 file split `.txt` + 6 file GT `.json` |
| 4 | `src/test_all.py` | **1** | Không | chỉ in ra màn hình: `58 PASS, 0 FAIL` |
| 5 | `src/self_check.py` | **5** | Có | chỉ in ra màn hình (1 lần kiểm môi trường + 1 lần mỗi method A/B/C/D) |
| 6 | `src/run_eval.py` | **35** | Có | 35 file JSON trong `work/results/` — mỗi file = 1 dòng bảng |
| 7 | `src/fill_excel.py` | **1** | Không | `docs/experiment.xlsx` được điền |

**35 lần `run_eval.py`** chia ra:

| Bảng | Sheet | Số lần | Chi tiết |
|---|---|---|---|
| Bảng 2 | `MAIN` | **15** | 4 phương pháp × các cặp (train, test): A 4 + C 4 + D 4 + B 3 |
| Bảng 1 | `VLM` | **20** | 5 model VLM × 4 tập test (`d1_train`, `d1_val`, `d2_train`, `d2_val`) |
| Bảng 3 | `quality` | **0** | không chạy thêm — lấy từ `by_quality` của 4 dòng đã có ở Bảng 2 |

Chỉ cần **4 lần train** cho cả 15 dòng Bảng 2, vì A / C / D dùng chung thành phần:

| Lần train | File | Train trên | Dòng nào dùng |
|---|---|---|---|
| 1 | `train.py` | D1 (MC-OCR) | A, C với `--train-on d1` |
| 2 | `train.py` | D2 (v3) | A, C với `--train-on d2` |
| 3 | `train_text_recognition.py` | D1 | A, D với `--train-on d1` |
| 4 | `train_text_recognition.py` | D2 | A, D với `--train-on d2` |

B không train gì cả (zero-shot).

---

## 2. Bốn bước chạy

```
build_gt.py  →  self_check.py  →  run_eval.py (nhiều lần)  →  fill_excel.py
  tạo GT        kiểm tra 3 ảnh      mỗi lần = 1 dòng bảng      điền Excel
```

### Bước 1 — `python build_gt.py`

Chạy **một lần**, không cần GPU, ~30 giây.

Đọc `mcocr_train_df.csv` + các file label JSON của v3, áp 5 rule chuẩn hóa,
sinh ground truth theo một schema chung.

**Output:**

| File | Nội dung |
|---|---|
| `work/splits/{d1_train,d1_val,d1_all,d2_train,d2_val,d2_all}.txt` | danh sách tên ảnh mỗi split, mỗi dòng 1 tên |
| `work/gt/d1_train.json` | 921 doc |
| `work/gt/d1_val.json` | 230 doc |
| `work/gt/d1_all.json` | 1.151 doc |
| `work/gt/d2_train.json` | 3.990 doc |
| `work/gt/d2_val.json` | 1.005 doc |
| `work/gt/d2_all.json` | 4.995 doc |

Mỗi doc là một dict:

```json
{
  "img_id": "mcocr_public_145013ahfqj",
  "quality": 0.63,
  "header":  {"merchant": "MINIMART ANAN", "address": "Chợ Sủi, ...",
              "date": "11/08/2020", "time": "08:06", "phone": null, ...},
  "line_items": null,
  "payment": {"total": "74,000", "discount": null, ...}
}
```

Hai điều cần chú ý trong output màn hình:

- **`doc 1155 dòng CSV -> 1153 doc hợp lệ`** — 2 ảnh bị loại vì `anno_num > 12`
  (nhãn lệch). Đây là chủ ý, không phải lỗi.
- **`line_items: null` với toàn bộ D1** — MC-OCR **không có nhãn ITEM_\***.
  `null` nghĩa là *bộ data không có nhãn* → bảng ghi `N/A`.
  Khác hẳn `[]` nghĩa là *có nhãn, hóa đơn này không có dòng hàng* → tính điểm 0.
  Phân biệt này chạy xuyên suốt code; đừng gộp hai thứ lại.

Sau đó chạy `python test_all.py` → phải ra `58 PASS, 0 FAIL`, exit 0.

### Bước 2 — `python self_check.py --method A --n 3`

**Bắt buộc trước mỗi phương pháp mới.** Không ghi file gì cả, chỉ in ra màn hình.

```bash
python self_check.py                                  # chỉ kiểm tra path + GT + thư viện
python self_check.py --method A --n 3                  # thử A trên 3 ảnh D1
python self_check.py --method B --n 2 --test-on d2_val # thử B trên 2 ảnh D2
```

In ra, cho từng ảnh: output thô của model, GT tương ứng, và điểm từng trường.
Bạn **đọc bằng mắt** xem có hợp lý không. Thoát 0 = ổn, khác 0 = đừng chạy tiếp.

Những gì nó bắt được: sai đường dẫn model, sai tên key trong dict mà PaddleOCR trả về,
VLM sinh ra text không phải JSON, crop rỗng, nhãn layout không khớp `LABEL2FIELD`.

### Bước 3 — `python run_eval.py ...`

**Một lần chạy = đúng một dòng trong bảng.** Chạy lại một dòng không ảnh hưởng dòng khác.

```bash
python run_eval.py --method A --train-on d2 --test-on d2_val
python run_eval.py --method A --train-on d2 --test-on d1_all
python run_eval.py --method A --train-on d2 --test-on d1_val
python run_eval.py --method A --train-on d1 --test-on d1_val
```

Tham số:

- `--method` — `A` / `B` / `C` / `D`, hoặc 5 biến thể VLM cho Bảng 1
  (`B_paddleocr_vl`, `B_qwen3vl_2b`, `B_qwen3vl_4b`, `B_qwen3vl_8b`, `B_internvl35_2b`),
  hoặc `dummy_perfect` / `dummy_noisy` / `dummy_empty` / `dummy_fail` để thử metric mà không cần model.
- `--train-on` — `d1` / `d2` / bỏ trống. Chỉ là **nhãn ghi vào kết quả** để `fill_excel.py`
  biết đặt vào dòng nào; model thật được chọn qua `VIE_LAYOUT_DIR` / `VIE_REC_DIR`.
  **Đổi checkpoint thì phải đổi biến môi trường, `--train-on` không tự làm việc đó.**
- `--test-on` — `d1_train` / `d1_val` / `d1_all` / `d2_train` / `d2_val`
- `--limit N` — chỉ chạy N ảnh, **lấy mẫu ngẫu nhiên seed 42** (không phải N ảnh đầu —
  lấy N đầu sẽ toàn template `cafe_*` và làm lệch cột đơn giá).
- `--save-pred` — lưu thêm output từng ảnh để debug.

**Output: một file JSON** tại `work/results/{method}__{train_on}__{test_on}.json`, gồm:

| Khóa | Nội dung |
|---|---|
| `method`, `train_on`, `test_on`, `n_images` | định danh dòng bảng |
| `parse_fail` | % ảnh model trả về không đọc được JSON |
| `fps`, `seconds_total` | tốc độ |
| `params_M`, `gflops`, `vram_mb` | đọc từ `work/logs/cost.json` (xem mục 4) |
| `layout_mAP50`, `layout_mAP50_95`, `rec_acc`, `rec_ned` | đọc từ `work/logs/subtask.json` |
| `agg.MFS` | **con số headline** — trung bình điểm các trường |
| `agg.per_field` | điểm từng trường: merchant, address, date, time, total |
| `agg.n_per_field` | **mẫu số riêng từng trường** — không phải trường nào cũng có nhãn ở mọi ảnh |
| `agg.halluc` | % trường model bịa ra trong khi GT là null |
| `items` | item F1 / name / qty / price / amount + mẫu số `n_*`. `null` nếu bộ test không có nhãn item |
| `by_quality` | cùng bộ metric, chia 3 tầng `clear` / `mid` / `blur` theo `anno_image_quality`. `null` với D2 (không có trường này) |
| `per_image` | điểm từng ảnh — dùng để debug và vẽ bảng chất lượng |

Với `--save-pred`: thêm `work/pred/{method}__{train_on}__{test_on}/{tên_ảnh}.json`.

In ra màn hình cuối mỗi lần chạy: MFS, Halluc, parse_fail, FPS, per_field, n_per_field.

### Bước 4 — `python fill_excel.py`

```bash
python fill_excel.py --dry-run     # xem sẽ điền gì, KHÔNG ghi
python fill_excel.py               # điền vào docs/experiment.xlsx
python fill_excel.py --out ../docs/experiment_filled.xlsx   # giữ bản gốc
```

Quét toàn bộ `work/results/*.json`, map vào đúng ô của 3 sheet.

- Chỉ ghi vào các ô đã khai báo trong `MAIN_ROWS` / `VLM_MODEL_ROWS` / `QUALITY_ROWS`.
  Không chạm nhãn, tiêu đề, màu, merge cell.
- Ô chưa có kết quả thì **giữ nguyên**, không xóa thành trống.
- Giá trị `None` (không áp dụng) ghi chuỗi **`N/A`**, không ghi `0`.
  Ghi `0` sẽ bị đọc là "làm được nhưng tệ", khác hẳn "không làm được".
- `ΔMFS` được **tính tự động** = MFS dòng cross-domain − MFS dòng in-domain (`d2`→`d2_val`).
- Nếu file đang mở trong Excel → báo rõ và thoát 2. Đóng Excel rồi chạy lại.

**Output:** `docs/experiment.xlsx` được cập nhật, và một log trên màn hình liệt kê
từng dòng đã điền kèm tên file nguồn.

---

## 3. Chạy gì để đầy cả 3 bảng

Mấu chốt: **chỉ cần 4 lần train**, vì A / C / D dùng chung thành phần.

| Lần train | Train gì | Dùng cho |
|---|---|---|
| 1 | layout detector trên D1 | A, C (dòng `train=d1`) |
| 2 | layout detector trên D2 | A, C (dòng `train=d2`) |
| 3 | text recognition trên D1 | A, D (dòng `train=d1`) |
| 4 | text recognition trên D2 | A, D (dòng `train=d2`) |

B không train gì cả.

### Bảng 2 — sheet `MAIN`, 15 dòng

```bash
# --- A (dòng 6-9) ---
export VIE_LAYOUT_DIR=.../layout_d2/inference ; export VIE_REC_DIR=.../rec_d2/inference
python run_eval.py --method A --train-on d2 --test-on d2_val    # dòng 6  in-domain
python run_eval.py --method A --train-on d2 --test-on d1_all    # dòng 7  cross-domain
python run_eval.py --method A --train-on d2 --test-on d1_val    # dòng 8  cross-domain
export VIE_LAYOUT_DIR=.../layout_d1/inference ; export VIE_REC_DIR=.../rec_d1/inference
python run_eval.py --method A --train-on d1 --test-on d1_val    # dòng 9  in-domain D1

# --- C (10-13), D (14-17): y hệt, đổi --method ---
# --- B (18-20): không cần VIE_LAYOUT_DIR / VIE_REC_DIR ---
python run_eval.py --method B --test-on d2_val                  # dòng 18
python run_eval.py --method B --test-on d1_all                  # dòng 19
python run_eval.py --method B --test-on d1_val                  # dòng 20
```

**Nhớ đổi `VIE_LAYOUT_DIR` / `VIE_REC_DIR` khi đổi `--train-on`.** Đây là chỗ dễ sai
nhất và sai thì không có gì báo — bảng vẫn đầy số, chỉ là số sai.

Ý nghĩa: dòng `train=d2, test=d2_val` là in-domain (hóa đơn tổng hợp → hóa đơn tổng hợp).
Dòng `train=d2, test=d1_*` là cross-domain — **câu hỏi chính của đồ án: train trên hóa
đơn sinh tự động thì nhận diện được hóa đơn thật không?** `ΔMFS` chính là khoảng cách đó.

### Bảng 1 — sheet `VLM`, 5 model × 4 tập test

```bash
for M in B_paddleocr_vl B_qwen3vl_2b B_qwen3vl_4b B_qwen3vl_8b B_internvl35_2b; do
  for T in d1_train d1_val d2_train d2_val; do
    python run_eval.py --method $M --test-on $T
  done
done
```

20 lần chạy. Block trên của sheet = split Train, block dưới = split Val.
Bảng này trả lời: VLM nào, bao nhiêu tham số, là đủ.

### Bảng 3 — sheet `quality`

**Không cần chạy thêm gì.** Lấy từ `by_quality` của 4 dòng đã có
(`A/C/D --train-on d1 --test-on d1_val` và `B --test-on d1_val`).
Chỉ D1 có `anno_image_quality` nên bảng này chỉ tồn tại trên D1 (12.6% ảnh có q < 0.5).

---

## 4. Hai file bạn phải tự ghi

Code **không bịa số**. Hai nhóm chỉ số dưới đây không đo được trong lúc inference,
nên `run_eval.py` đọc từ file. Chưa có file → bảng hiện `N/A`.

`work/logs/subtask.json` — kết quả đánh giá 2 sub-task, lấy từ log khi bạn train:

```json
{
  "layout_d1": {"mAP50": 0.912, "mAP50_95": 0.701},
  "layout_d2": {"mAP50": 0.0,   "mAP50_95": 0.0},
  "rec_d1":    {"acc": 71.85,   "ned": 0.9398},
  "rec_d2":    {"acc": 0.0,     "ned": 0.0}
}
```

`work/logs/cost.json` — Params / GFLOPs, đo bằng `paddle.flops` hoặc `thop`:

```json
{
  "A": {"params_M": 0.0, "gflops": 0.0, "vram_mb": 0},
  "B": {"params_M": 0.0, "gflops": 0.0, "vram_mb": 0},
  "C": {},
  "D": {},
  "B_qwen3vl_4b": {}
}
```

(Các số `0.0` ở trên chỉ là **chỗ trống minh họa định dạng** — thay bằng số bạn đo được.
Đừng để nguyên.)

---

## 5. Lỗi hay gặp

| Hiện tượng | Nguyên nhân |
|---|---|
| `CANH BAO: khong nap duoc methods/real.py` | chưa cài `paddleocr` / `transformers`, hoặc `real.py` có lỗi. Dòng tiếp theo in đúng exception. |
| `chua dang ky method 'A'` | như trên — `real.py` không nạp được nên A chưa đăng ký |
| `RO RI SPLIT: N anh nam o ca train va val` | file `text_recognition_*_data.txt` bị sửa. Split D1 **phải** lấy từ 2 file này: model recognition đã train trên đúng các crop đó, chia lại seed mới sẽ rò rỉ và đẩy điểm A lên sai. |
| Toàn bộ `item_*` ra `N/A` khi test trên D1 | đúng như vậy. D1 không có nhãn `ITEM_*`. |
| `price` thấp bất thường trên D2 | kiểm tra `n_price` trong JSON. Nhiều template cafe không in đơn giá; các món đó bị loại khỏi mẫu số, không tính là sai. |
| `fill_excel.py` báo `PermissionError` | file đang mở trong Excel |
| `fill_excel.py`: `khong co o nao khop` | `--method` / `--train-on` / `--test-on` không đúng bộ khóa trong `MAIN_ROWS` |
| `OMP: Error #15 libiomp5md.dll` | xung đột OpenMP trên Windows. Đặt `KMP_DUPLICATE_LIB_OK=TRUE`. |
| Tiếng Việt ra `?????` trên console | đặt `PYTHONIOENCODING=utf-8` |

---

## 6. Bản đồ file

```
src/
  cfg.py              đường dẫn, TEST_SETS, ngưỡng chia tầng chất lượng
  schema.py           schema chung header / line_items / payment
  build_gt.py         BƯỚC 1
  self_check.py       BƯỚC 2
  run_eval.py         BƯỚC 3
  fill_excel.py       BƯỚC 4 — bản đồ ô Excel nằm ở đầu file
  test_all.py         58 test, không cần GPU
  data/
    splits.py         split D1 từ file recognition + assert chống rò rỉ
    gt_mcocr.py       5 rule chuẩn hóa GT cho MC-OCR
    gt_v3.py          gom ITEM_* theo khoảng Y
  metrics/
    core.py           ANLS / numeric_match / norm_date_em / exact + bắt buộc NFC
    fields.py         điểm từng trường, MFS, hallucination
    items.py          ghép món rồi tính F1, mẫu số riêng từng trường
  methods/
    base.py           interface Method + registry
    postproc.py       output thô -> doc chuẩn, DÙNG LẠI rule của GT
    real.py           A / B / C / D + 5 biến thể VLM   <-- phần chưa chạy thật
    dummy.py          4 method giả để thử metric mà không cần model
```

**Một điểm thiết kế đáng nhớ:** `methods/postproc.py` import trực tiếp `TS_PREFIX`
và `_is_amount` từ `data/gt_mcocr.py` thay vì copy lại. Nếu prediction không đi qua
đúng các bước chuẩn hóa mà GT đã đi qua, ta sẽ so `"Ngày : 11/08/2020"` với `"11/08/2020"`
rồi kết luận model đọc sai trong khi nó đọc đúng. Loại lỗi này không làm chương trình
crash nên rất khó phát hiện — nó chỉ làm toàn bộ bảng sai.
