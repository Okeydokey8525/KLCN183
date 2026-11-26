# Hướng 4, mốc 3: huấn luyện TSVM seed 0 và 15 bằng kernel CUDA

Seed 0 là mốc so sánh chính (báo cáo: 0,7245, epoch 88, 4,52 giờ). Seed 15 là lượt có mAP gần trung bình 20 seed, best.pt của nó dùng cho web (báo cáo: 0,7262, epoch 84, 4,73 giờ). Hai seed chạy lần lượt trong cùng một phiên.

Tạo ngày 24/11/2026 bằng [`../scripts/tao_goi_kaggle_h4_moc3.py`](../scripts/tao_goi_kaggle_h4_moc3.py).

Notebook: `3_notebooks/h4-moc3-huan-luyen-s0-s15.ipynb`.

Siêu tham số chép nguyên (106 tham số) từ `args.yaml` của lượt báo cáo `TSVM_FP32_s0_w2`. Chỉ khác `TSVM_SCAN=cuda`.

## Đợt 2 (25/11): 18 seed còn lại cho đủ 20 seed

Seed 0 và 15 đã chạy xong ở đợt 1 (mỗi seed 2,40 giờ). 18 seed còn lại chia thành 5 notebook:

| Notebook | Seed | Ước tính | Tệp tải về |
|---|---|---|---|
| `h4-moc3-huan-luyen-s1-s2.ipynb` | 1, 2 | khoảng 4,8 giờ | `h4_moc3_s1_s2.zip` |
| `h4-moc3-huan-luyen-s3-s4-s5-s6.ipynb` | 3–6 | khoảng 9,6 giờ | `h4_moc3_s3_s4_s5_s6.zip` |
| `h4-moc3-huan-luyen-s7-s8-s9-s10.ipynb` | 7–10 | khoảng 9,6 giờ | `h4_moc3_s7_s8_s9_s10.zip` |
| `h4-moc3-huan-luyen-s11-s12-s13-s14.ipynb` | 11–14 | khoảng 9,6 giờ | `h4_moc3_s11_s12_s13_s14.zip` |
| `h4-moc3-huan-luyen-s16-s17-s18-s19.ipynb` | 16–19 | khoảng 9,6 giờ | `h4_moc3_s16_s17_s18_s19.zip` |

Mỗi notebook chạy giống đợt 1 (Bước 2 bên dưới). Giá trị tham chiếu PyTorch của từng seed được tính tự động từ `results.csv` gốc.

**Lưu ý:**
- Notebook 4 seed chỉ còn dư khoảng 2 giờ so với giới hạn 12 giờ của Kaggle.
  - Nếu phiên bị cắt, các seed đã xong vẫn được giữ trong Output, nhưng cell 3 sẽ không chạy, nên không có tệp zip.
  - Khi đó báo mình để tạo notebook chạy lại các seed còn thiếu.
- Hạn mức GPU miễn phí của Kaggle khoảng 30 giờ mỗi tuần. 5 notebook cần khoảng 43 giờ, nên phải chạy trong 2 tuần, hoặc dùng thêm tài khoản khác.

## Bước 1. Tạo dataset cho wheel (một lần)

1. **Create → New Dataset**, kéo `mamba_ssm-2.3.2.post1-cp313-cp313-linux_x86_64.whl` (349 MB, đang ở Downloads) vào.
2. Đặt tên `h4-mamba-wheel`, chọn **Private**.

## Bước 2. Chạy notebook

1. Import `h4-moc3-huan-luyen-s0-s15.ipynb`.
2. **Add Input** hai dataset: `h4-tsvm-cuda` (đã có từ mốc 2) và `h4-mamba-wheel`.
3. Chọn **GPU T4 x2**, bật **Internet On** (để tải `yolo26s-seg.pt`).
4. Chọn **Save Version → Save & Run All (Commit)**, để tắt trình duyệt vẫn chạy tiếp.
5. Dự kiến khoảng 5 giờ cho hai seed (khoảng 2,5 giờ mỗi seed), dưới giới hạn 12 giờ của Kaggle. Đây là ước tính từ phép đo một bước ở mốc 2; bản PyTorch mất 4,52 giờ.

## Log cần có

| Cell | Dòng cần thấy |
|---|---|
| 0 | `✅ Có kernel selective_scan_cuda, mamba_ssm 2.3.2.post1` |
| 1 | `✅ Ultralytics 8.4.127 (Hướng 4), TSVM_SCAN=cuda` |
| 2 | Huấn luyện 100 epoch, cuối cùng in `⏱️ Huấn luyện ... giờ` |
| 3 | Bảng JSON có `mask_map50_95_tot_nhat`, `delta_map`, `ket_luan` |

## Bước 3. Tải kết quả

Ở tab Output, tải `h4_moc3_s0_s15.zip` (gồm thư mục lượt chạy: `results.csv`, `weights/best.pt`, `tom_tat_moc3.json`) rồi gửi mình.

## Nếu lỗi

| Thông báo | Cách xử lý |
|---|---|
| Cell 0 báo `undefined symbol` hoặc `⚠️ Kaggle đã đổi torch` | Wheel không khớp với bản torch mới của Kaggle. Gửi mình dòng lỗi; khi đó phải biên dịch lại (khoảng 50 phút) |
| `RuntimeError ... deterministic` | Gửi mình dòng lỗi |

## Lưu ý

- Kernel `mamba_ssm` cộng dồn gradient bằng atomicAdd, nên hai lần chạy cùng seed có thể lệch nhau rất nhỏ. Bản PyTorch thì lặp lại được tuyệt đối.
- Vì vậy so sánh theo quy tắc ±0,0074 (1 SD của 20 seed), không đòi giống từng bit.
