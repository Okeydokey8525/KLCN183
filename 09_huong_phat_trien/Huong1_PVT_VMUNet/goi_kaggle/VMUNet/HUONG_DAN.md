# Gói chạy VM-UNet trên các tài khoản Kaggle khác

Tạo ngày 16/11/2026. Giao thức giống PraNet/U-Net/Polyp-PVT 640: BG20, ảnh 640, 100 epoch, fp32, seed 0–4. Riêng VM-UNet dùng **batch 4 × cộng dồn 2 bước** (batch 8 hết bộ nhớ T4). Mô hình chỉ dùng LayerNorm nên cách này gần như tương đương batch 8.

Chạy thử 16/11 trên T4: 8,7 phút/epoch, 12,1 GB GPU, nên **mỗi seed khoảng 14,5 giờ (2 phiên)**.

## Trong thư mục này

| File | Dùng để |
|---|---|
| `1_dataset_bg20/Kvasir_YOLO_SEG_BG20.zip` (102 MB) | Dataset BG20 (train 880 + 160 nền, val 120 + 40 nền) |
| `2_dataset_vm_unet_code/vm-unet-code.zip` (679 MB) | Mã gốc VM-UNet + trọng số `vmamba_small_e238_ema.pth` |
| `3_notebooks/vm-unet-640-seed0.ipynb` … `seed4.ipynb` | Notebook chạy chính, đã đặt sẵn `SEED`, `CHAY_THU = False`, `BATCH_SIZE = 4`, `ACCUM = 2` |

**Thiếu wheel `mamba_ssm`.** Bản đã chạy được là `mamba_ssm-2.3.2.post1-cp313-cp313-linux_x86_64.whl` (349 MB). File này chỉ có trong dataset `okeydokey0805/vm-unet-code` của tài khoản chính, máy chưa có bản này. Vì vậy chọn **cách A** dưới đây.

## Bước 1. Chuẩn bị dataset cho tài khoản phụ

### Cách A (khuyên dùng): chia sẻ dataset của tài khoản chính

Không phải tải lên lại 1 GB, và dùng đúng wheel đã chạy thử.

1. Đăng nhập tài khoản chính `okeydokey0805`.
2. Mở dataset `vm-unet-code`, vào **Settings → Sharing**, thêm tên các tài khoản phụ, quyền **Viewer**. Không để Public.
3. Làm lại với dataset `kvasir-yolo-seg-bg20`.
4. Ở tài khoản phụ, khi **Add Input**, chọn tab **Shared with you** (hoặc dán đường dẫn dataset) để thêm 2 dataset này.

### Cách B: tải lên dataset riêng ở từng tài khoản

1. **Create → New Dataset**, kéo `Kvasir_YOLO_SEG_BG20.zip` vào, để **Private**, Create.
2. Tạo dataset thứ hai từ `vm-unet-code.zip`, **thêm cả file wheel `mamba_ssm-...whl`** vào cùng dataset, để **Private**.
3. Nếu không có wheel, notebook tự cài `mamba-ssm` từ PyPI (phải bật Internet). Mỗi phiên có thể mất thêm 30–60 phút để biên dịch, và có thể lỗi nếu Kaggle đổi phiên bản PyTorch.

Không cần giải nén zip. Notebook tự dò `Kvasir_YOLO_SEG_BG20`, `models/vmunet/vmunet.py`, `vmamba_small_e238_ema.pth` và `mamba_ssm-*.whl` ở bất kỳ đâu trong `/kaggle/input`.

## Chia tài khoản

Mỗi seed khoảng 14,5 giờ GPU. Mỗi tài khoản phải **còn ít nhất 15 giờ GPU** trong tuần và chỉ chạy **1 seed**.

| Tài khoản | Seed |
|---|---|
| … | 0 |
| … | 1 |
| … | 2 |
| … | 3 |
| … | 4 |
| Dự phòng | chạy lại seed bị lỗi |

## Bước 2. Tạo notebook

1. **Create → New Notebook → File → Import Notebook**, chọn `vm-unet-640-seed<N>.ipynb` đúng seed của tài khoản.
2. **Add Input**: thêm 2 dataset (BG20 và `vm-unet-code`).
3. **Settings → Accelerator → GPU T4 x2**. Notebook tự chỉ dùng 1 GPU.
4. **Settings → Internet → On**. Cần bật vì có thể phải cài gói.
5. Mở Cell 0 kiểm tra: `SEED = <N>`, `CHAY_THU = False`, `BATCH_SIZE = 4`, `ACCUM = 2`.

## Bước 3. Chạy phiên 1

Bấm **Save Version → Save & Run All (Commit) → Save**. Có thể tắt trình duyệt.

Đầu log phải có các dòng sau:
- `✅ Kernel selective_scan chạy được`
- `Encoder nhận 117 tensor, decoder nhận 91 tensor`
- `Số tham số: 27,427,561`

Notebook dừng ở khoảng 11 giờ (khoảng epoch 75) và in `⏸ Dừng ở epoch ... Đã lưu resume.pth`.

**Nếu bị Queued quá 15–20 phút:** chạy tương tác.
1. Bấm **Edit**, rồi **Run All**, để trang mở suốt phiên.
2. Cảnh báo "unable to save" không ảnh hưởng việc chạy.
3. Khi xong, **Save Version** với tùy chọn **Quick Save** để giữ Output (có `resume.pth`).

## Bước 4. Chạy phiên 2 (chạy tiếp)

1. Mở lại notebook (**Edit**).
2. **Add Input → Your Work**, chọn **chính notebook này** (output của phiên 1).
3. Không sửa gì khác, giữ nguyên `SEED`.
4. **Save Version → Save & Run All**.
5. Log phải in `↻ Chạy tiếp từ epoch ...`. Nếu lại chạy từ epoch 1 thì chưa gắn output phiên 1: dừng lại và kiểm tra Input.

Phiên 2 khoảng 3,5–4 giờ.

## Bước 5. Tải kết quả về

Khi log in `Xong. Val Dice tốt nhất ...`:
1. Vào tab **Output** của version cuối, tải `runs_VMUNet_640_aug_s<N>.zip`.
2. Giải nén vào `Baseline_PVT_VMUNet/ket_qua/VMUNet_640/`.

Trong đó có `results.csv`, `tom_tat.txt` và checkpoint tốt nhất `vmunet_best_s<N>.pth`.

Đánh giá theo pixel phải chạy trên Kaggle, vì `mamba_ssm` không cài được trên Windows. Notebook đánh giá sẽ làm sau khi có checkpoint.

## Lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `❌ Chưa bật GPU` | Settings → Accelerator → GPU T4 x2 |
| `❌ Chưa thêm dataset chứa mã VM-UNet` hoặc thiếu `vmamba_small_e238_ema.pth` | Thiếu Input `vm-unet-code` |
| Lỗi khi cài hoặc import `mamba_ssm`, `selective_scan` | Dataset thiếu wheel và không bật Internet. Dùng cách A, hoặc bật Internet |
| CUDA out of memory | Không xảy ra ở batch 4 khi chạy thử (12,1 GB). Kiểm tra Cell 0 đúng `BATCH_SIZE = 4` |
| Queued lâu | Chạy tương tác như Bước 3 |

## Lưu ý

- Không đổi `SEED` sau khi đã xem kết quả, không chạy thêm seed ngoài 0–4.
- Không để dataset Public.
