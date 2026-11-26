# Gói chạy Polyp-PVT trên các tài khoản Kaggle khác

Tạo ngày 15/11/2026. Giao thức giống PraNet/U-Net 640: BG20, ảnh 640, 100 epoch, batch 8, fp32, seed 0–4.

## Trong thư mục này

| File | Dùng để |
|---|---|
| `1_dataset_bg20/Kvasir_YOLO_SEG_BG20.zip` (99 MB) | Dataset BG20 (train 880 + 160 nền, val 120 + 40 nền) |
| `2_dataset_polyp_pvt_code/polyp-pvt-code.zip` (92 MB) | Mã gốc Polyp-PVT + trọng số `pvt_v2_b2.pth` |
| `3_notebooks/polyp-pvt-640-seed0.ipynb` … `seed4.ipynb` | Notebook chạy chính, mỗi file đã đặt sẵn `SEED` |

Không cần giải nén zip. Kaggle tự giải nén khi tạo dataset, và notebook tự dò thư mục `Kvasir_YOLO_SEG_BG20`, `lib/pvt.py`, `pvt_v2_b2.pth` ở bất kỳ đâu trong `/kaggle/input`, nên đặt tên dataset thế nào cũng được.

## Chia tài khoản (tuần 15–20/11)

Mỗi seed khoảng 14,5 giờ GPU (2 phiên), nên mỗi tài khoản chỉ chạy **1 seed** trong tuần để không vượt 30 giờ.

| Tài khoản | Seed | Ghi chú |
|---|---|---|
| okeydokey0805 (chính) | 0 | Đã có sẵn: notebook `notebook42ed41b80e` |
| Tài khoản 2 | 1 | |
| Tài khoản 3 | 2 | |
| Tài khoản 4 | 3 | |
| Tài khoản 5 | 4 | |
| Tài khoản 6 | dự phòng | Chạy lại seed bị lỗi, hoặc thử VM-UNet |

## Các bước trên mỗi tài khoản

### Bước 1. Tạo 2 dataset (làm một lần mỗi tài khoản)

1. Kaggle → **Create → New Dataset** → kéo file `Kvasir_YOLO_SEG_BG20.zip` vào → đặt tên (ví dụ `kvasir-yolo-seg-bg20`) → để **Private** → Create.
2. Làm lại với `polyp-pvt-code.zip` (tên ví dụ `polyp-pvt-code`), cũng để **Private**. Không để Public, vì Polyp-PVT chỉ cho dùng vào nghiên cứu, giáo dục.
3. Chờ cả hai dataset báo xử lý xong.

### Bước 2. Tạo notebook

1. **Create → New Notebook → File → Import Notebook** → chọn file `polyp-pvt-640-seed<N>.ipynb` đúng seed của tài khoản đó.
2. Bên phải: **Add Input** → tab **Your Work** → thêm 2 dataset vừa tạo.
3. **Settings → Accelerator → GPU T4 x2**. Notebook tự chỉ dùng 1 GPU, giống lượt PraNet.
4. Mở Cell 0 kiểm tra: `SEED = <N>` đúng, `CHAY_THU = False`.

### Bước 3. Chạy phiên 1

Bấm **Save Version → Save & Run All (Commit) → Save**. Có thể tắt trình duyệt, notebook chạy nền.

Notebook dừng ở khoảng 11 giờ (khoảng epoch 75), log in `⏸ Dừng ở epoch ... Đã lưu resume.pth`. Kaggle cắt phiên ở 12 giờ, nên phải dừng sớm để giữ được kết quả.

### Bước 4. Chạy phiên 2 (chạy tiếp)

1. Mở lại notebook (**Edit**).
2. **Add Input** → **Your Work** → chọn **notebook này** (output của phiên 1).
3. **Không sửa gì khác**, giữ nguyên `SEED`.
4. **Save Version → Save & Run All**. Log phải in `↻ Chạy tiếp từ epoch ...`. Nếu không thấy dòng này mà lại chạy từ epoch 1, tức là chưa gắn output phiên 1: dừng lại và kiểm tra Input.

### Bước 5. Tải kết quả về

Khi log in `Xong. Val Dice tốt nhất ...`:
1. Tab **Output** của version cuối → tải `runs_PolypPVT_640_aug_s<N>.zip`.
2. Đổi tên nếu cần, gửi về và giải nén vào `Baseline_PVT_VMUNet/ket_qua/PolypPVT_640/`.

Trong đó có `results.csv`, `tom_tat.txt` và checkpoint tốt nhất `polyp_pvt_best_s<N>.pth`.

## Lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `❌ Chưa bật GPU` | Settings → Accelerator → GPU T4 x2, rồi chạy lại |
| `❌ Chưa thêm dataset BG20` | Thiếu Input BG20, hoặc dataset chưa xử lý xong |
| Không tìm thấy `lib/pvt.py` hoặc `pvt_v2_b2.pth` | Thiếu Input `polyp-pvt-code` |
| `Backbone nhận .../332` thấp, bị dừng | File trọng số hỏng, tải lại `polyp-pvt-code.zip` |
| Xếp hàng (Queued) quá lâu | Kaggle thiếu GPU giờ cao điểm. Cứ để chờ, hoặc dừng và chạy lại lúc khác. Lượt xếp hàng không tốn giờ GPU |
| CUDA out of memory | Không xảy ra ở batch 8 khi chạy thử (12,8 GB / 15 GB). Nếu có: sửa `BATCH_SIZE = 4`, `ACCUM = 2` |

## Lưu ý

- Không đổi `SEED` sau khi đã xem kết quả, không chạy thêm seed ngoài 0–4.
- Chạy thử trên T4: 8,7 phút/epoch, 12,8 GB GPU.
