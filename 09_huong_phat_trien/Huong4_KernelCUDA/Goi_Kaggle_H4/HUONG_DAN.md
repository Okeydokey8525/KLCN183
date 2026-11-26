# Hướng 4, mốc 2: kiểm tra kernel CUDA và đo tốc độ trên Kaggle

Tạo ngày 24/11/2026 bằng [`../scripts/tao_goi_kaggle_h4.py`](../scripts/tao_goi_kaggle_h4.py). Mốc này **chưa huấn luyện**. Nó chỉ trả lời hai câu hỏi:
1. Kernel `selective_scan` thật có cho kết quả giống bản PyTorch không?
2. Nhanh hơn bao nhiêu?

## Trong thư mục này

| File | Dùng để |
|---|---|
| `1_dataset/h4-tsvm-cuda.zip` (176 MB) | Mã Hướng 4 (`ultralytics_TSVM_CUDA/`), dữ liệu BG20 (train 1.040 / val 160, dùng lại ở mốc 3), `best.pt` seed 0 của báo cáo (baseline và TSVM), `ma_kiem_tra/h4_kiem_tra.py` |
| `3_notebooks/h4-moc2-kiem-tra-kernel.ipynb` | Cài `mamba_ssm`, chạy phần A–F, nén kết quả |
| `ma_kiem_tra/h4_kiem_tra.py` | Bản gốc của tệp kiểm tra nằm trong zip. Đã chạy thử trên laptop bằng kernel giả lập |

## Bước 1. Tạo dataset

1. **Create → New Dataset**, kéo `h4-tsvm-cuda.zip` vào.
2. Đặt tên `h4-tsvm-cuda`, chọn **Private**, bấm Create. Không cần giải nén.

## Bước 2. Chạy notebook (tương tác hoặc Save & Run All đều được)

1. **Create → New Notebook → File → Import Notebook**, chọn `h4-moc2-kiem-tra-kernel.ipynb`.
2. **Add Input** dataset `h4-tsvm-cuda`.
3. **Settings → Accelerator → GPU T4 x2**, bật **Internet → On** (bắt buộc, để cài `mamba-ssm`).
4. **Run All**, hoặc **Save Version → Save & Run All (Commit)**.

**Thời gian dự kiến:**
- Phần A–F mất khoảng 15–25 phút.
- Cài `mamba-ssm` mất vài phút nếu có bản dựng sẵn khớp PyTorch của Kaggle. Nếu không có, gói phải tự biên dịch, mất 20–60 phút.

## Log cần có

| Cell | Dòng cần thấy |
|---|---|
| 1 | `✅ Có kernel selective_scan_cuda, mamba_ssm ...` |
| 2 | `✅ Ultralytics 8.4.127 (bản Hướng 4) ...; kernel: có` |
| 3, phần A | `A  sai khác lớn nhất: ... -> PASS` |
| 3, phần B | `B  \|mAP50-95 PyTorch - kernel\| = ... -> PASS` |
| 3, phần C–F | Bảng `C ... tổng hợp`, `D khối SS2D`, `E tổng hợp`, `F tổng hợp` |
| 4 | `Tải về: ['h4_moc2_ket_qua.zip'] và ['mamba_ssm-....whl']` |

## Phần A–F đo gì

| Phần | Nội dung |
|---|---|
| A | Kernel thật so với PyTorch: SS2D ngẫu nhiên (forward và gradient), SS2D và nhánh TSVMamba của tầng 10 với đặc trưng thật của 8 ảnh val |
| B | Đánh giá 160 ảnh val bằng `best.pt` seed 0: baseline, TSVM-PyTorch, TSVM-kernel |
| C | Độ trễ forward, batch 1, FP32, chỉ tính mô hình. 5 vòng xen kẽ, mỗi vòng 200 lần, đổi thứ tự cấu hình mỗi vòng |
| D | Độ trễ riêng khối SS2D, batch 1 và batch 8 |
| E | Một bước huấn luyện (batch 8, forward + backward + AdamW) và bộ nhớ GPU đỉnh |
| F | Giả lập video: dự đoán lần lượt 160 ảnh val như các khung hình liên tiếp, tính cả tiền xử lý và hậu xử lý, đo số khung/giây |

## Bước 3. Tải kết quả

Ở tab **Output**, tải hai thứ về một chỗ rồi báo mình:
- `h4_moc2_ket_qua.zip`
- `wheels/mamba_ssm-*.whl`: giữ lại để mốc 3 khỏi phải biên dịch lại.

## Nếu lỗi

| Thông báo | Cách xử lý |
|---|---|
| `❌ Không cài được kernel mamba_ssm` | Tải `cai_mamba_ssm_log.txt` ở Output gửi mình |
| `❌ Chưa thêm dataset h4-tsvm-cuda` | Thiếu Add Input |
| `❌ Mã Hướng 4 không nạp được kernel` | Cell 1 đã cài nhưng import lỗi. Gửi mình dòng lỗi đi kèm |
| Phần A hoặc B in `CẦN XEM LẠI` | Không cần dừng, cứ chạy hết rồi gửi kết quả |

## Lưu ý

- Mốc này không huấn luyện và không dùng tập test.
- Số đo trên T4 không so trực tiếp được với số 39,86 ms đo trên RTX 5050 của báo cáo, nhưng **tỉ lệ giữa các cấu hình** thì so được. Baseline được đo cùng phiên để làm mốc.
- Không để dataset Public.
