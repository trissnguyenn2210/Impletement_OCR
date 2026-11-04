# Web demo — Receipt VIE (4 phương pháp)

Giao diện web (Gradio) để chạy **A / B / C / D** trên một ảnh hoá đơn và so sánh kết quả.

| Method | Pipeline | Model dùng |
|---|---|---|
| **A** OCR pipeline | layout → crop → text det → CTC rec → gán trường theo nhãn layout | layout + rec **fine-tune** (`checkpoints/`), det `PP-OCRv5_server_det` (tự tải) |
| **B** VLM zero-shot | cả trang ảnh → VLM → JSON | VLM chọn trên giao diện (mặc định `Qwen/Qwen3-VL-2B-Instruct`) |
| **C** Layout + VLM | layout → crop → VLM đọc từng vùng | layout fine-tune + VLM |
| **D** OCR + LLM | det + rec cả trang → LLM (chỉ thấy text) → JSON | rec fine-tune + LLM (mặc định `Qwen/Qwen3-4B-Instruct-2507`) |

Code model nằm ở `src/methods/real.py` (bản đã sửa lỗi). Demo dùng chung backend: A và C dùng chung
layout, A và D dùng chung OCR, nên mỗi model chỉ nạp một lần.

## 1. Checkpoint

```
checkpoints/
├── layout/inference/        inference.yml, inference.pdiparams, inference.json
└── recognition/inference/   inference.yml, inference.pdiparams, inference.json
```

Demo tự trỏ `VIE_LAYOUT_DIR` / `VIE_REC_DIR` vào hai thư mục trên. Muốn dùng checkpoint khác thì set
biến môi trường trước khi chạy.

## 2. Cài đặt

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r demo/requirements.txt
```

GPU: cài `paddlepaddle-gpu` và torch bản CUDA thay cho bản CPU (xem chú thích trong `demo/requirements.txt`).

## 3. Chạy

```bash
python demo/app.py                    # mở http://127.0.0.1:7860
python demo/app.py --share            # thêm link public *.gradio.live
python demo/app.py --device cpu       # ép chạy CPU
python demo/app.py --vlm Qwen/Qwen3-VL-4B-Instruct --llm Qwen/Qwen3-4B-Instruct-2507
python demo/app.py --low-mem          # giải phóng model sau mỗi phương pháp (GPU ít VRAM)
python demo/app.py --max-new-tokens 512
```

Trên giao diện: tải ảnh (hoặc bấm ảnh mẫu MC-OCR) → tick phương pháp → **Chạy**.
Tab **So sánh** hiện các trường của mọi phương pháp cạnh nhau; mỗi tab A/B/C/D có ảnh vẽ vùng
layout / dòng OCR, bảng trường, line items, JSON chuẩn và trace (text từng vùng, output thô VLM/LLM).

## 4. Phần cứng — lưu ý

- Thiết bị tự nhận: Paddle dùng `gpu:0` nếu cài bản GPU, torch dùng `cuda:0` nếu có CUDA; còn lại chạy CPU.
  GPU không hỗ trợ BF16 (GTX 16xx / RTX 20xx) tự chuyển sang FP16.
- **A** chạy được trên CPU (~15 s/ảnh, lần đầu nạp model ~1 phút).
- **B / C / D** dùng VLM/LLM vài tỷ tham số: Qwen3-VL-2B cần ~5 GB VRAM, 4B ~9 GB, 8B ~17 GB (BF16).
  Trên CPU chạy được nhưng chậm (vài phút/ảnh). **C** gọi VLM cho từng vùng layout nên chậm hơn B.
- Lần đầu chọn VLM/LLM, model sẽ được tải từ HuggingFace (vài GB).

Các biến môi trường của `src/methods/real.py` vẫn dùng được: `VIE_PADDLE_DEVICE`, `VIE_TORCH_DEVICE`,
`VIE_TORCH_DTYPE` (bfloat16/float16/float32), `VIE_PADDLE_MKLDNN=1` (bật lại oneDNN trên CPU),
`VIE_QWEN_MAX_PIXELS`, `VIE_ATTN_IMPL`, …
