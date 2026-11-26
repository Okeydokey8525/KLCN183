# VM-UNet 640, seed 0–4: kết quả

Huấn luyện trên Kaggle (Tesla T4) xong ngày 17/11/2026. Đánh giá trên laptop cùng ngày. Kết quả chỉ dùng cho slide và phần trả lời hội đồng, không sửa báo cáo đã nộp.

## 1. Kiểm tra lượt huấn luyện

| Kiểm tra | Kết quả |
|---|---|
| Đủ 5 seed, mỗi seed 100 epoch, epoch liên tục 1–100 | Đạt. `tom_tat.txt` ghi `xong=True resumed=True` (mỗi seed chạy 2 phiên) |
| Chỗ nối 2 phiên | Cột thời gian tăng đều, không bị đặt lại. Tốc độ học nối tiếp đúng lịch |
| Giao thức | BG20, ảnh 640, 100 epoch, fp32, tăng cường giống YOLO, batch 4 × cộng dồn 2 (≈ batch 8, mô hình chỉ có LayerNorm) |
| Môi trường | torch 2.11 + cu130, mamba_ssm 2.3.2.post1, Tesla T4, VRAM tối đa 12,1 GB |
| Số tham số | 27.427.561 |
| Thời gian | 14,4 ± 0,4 giờ/seed (Polyp-PVT: 13,9 giờ) |

| Seed | Val Dice tốt nhất (notebook) | Epoch tốt nhất | Giờ huấn luyện |
|---:|---:|---:|---:|
| 0 | 0,7543 | 68 | 15,1 |
| 1 | 0,7587 | 97 | 14,0 |
| 2 | 0,7209 | 98 | 14,6 |
| 3 | 0,7976 | 97 | 14,1 |
| 4 | 0,8007 | 85 | 14,5 |
| TB | 0,766 ± 0,033 | | 14,4 |

**Hai điểm bất thường (đã xác minh nguyên nhân):**
- **Tốc độ học tăng lại sau epoch 50.** Notebook chép đúng cấu hình gốc của VM-UNet: Cosine `T_max = 50`, `eta_min = 1e-5` (`configs/config_setting.py`). Bản gốc huấn luyện 300 epoch nên tốc độ học đi lên xuống nhiều chu kỳ. Ở 100 epoch, tốc độ học giảm đến epoch 50 rồi tăng dần về 1e-3 ở epoch 100.
- **Seed 0 phân kỳ ở epoch 86–90**, khi tốc độ học gần tối đa: loss tăng từ 0,63 lên 1,08, Val Dice tụt xuống khoảng 0,3. Checkpoint tốt nhất của seed 0 là epoch 68, trước lúc phân kỳ, nên kết quả đánh giá không bị ảnh hưởng.
- 3/5 seed đạt tốt nhất ở epoch 97–98, nên mô hình có thể chưa hội tụ hẳn.

## 2. Đánh giá theo pixel (giao thức Bảng 4.8)

**Cách chạy trên Windows.** `mamba_ssm` không cài được trên Windows. Mã gốc gọi phép quét qua tên toàn cục `selective_scan_fn` (import nằm trong try/except), nên mình thay tên này bằng bản viết thuần PyTorch: [`danh_gia/vmunet_quet_thuan.py`](../../danh_gia/vmunet_quet_thuan.py).
- So với bản quét tuần tự trên dữ liệu ngẫu nhiên: lệch khoảng 1e-6.
- **Kiểm chứng đầu-cuối:** từ cùng lần suy luận, tính lại Val Dice đúng cách notebook. Cả 5 seed lệch so với `tom_tat.txt` dưới 1e-6 ([`13_kiem_chung_val_dice_vmunet.csv`](../danh_gia_pixel/13_kiem_chung_val_dice_vmunet.csv)). Như vậy mô hình dựng trên laptop cho đúng kết quả của bản chạy trên Kaggle.
- Suy luận khoảng 1,2 giây/ảnh (phép quét thuần PyTorch chậm). Con số này **không phải độ trễ thật** của VM-UNet.

Script: [`danh_gia/danh_gia_pixel_vmunet.py`](../../danh_gia/danh_gia_pixel_vmunet.py). Giao thức giống Polyp-PVT: logit phóng song tuyến về kích thước gốc, ngưỡng 0,5, nhãn polygon, 120 ảnh polyp và 40 ảnh nền.

| Mô hình | Dice | IoU | Precision | Recall | Ảnh nền báo nhầm |
|---|---:|---:|---:|---:|---:|
| Polyp-PVT | 0,894 ± 0,005 | 0,831 | 0,909 | 0,913 | 3,0% |
| PraNet | 0,885 ± 0,005 | 0,821 | 0,899 | 0,911 | 2,5% |
| YOLO26s-seg | 0,854 ± 0,022 | 0,798 | 0,867 | 0,873 | 0,5% |
| YOLO26s-seg + TSVM | 0,854 ± 0,011 | 0,800 | 0,858 | 0,877 | 2,0% |
| **VM-UNet** | **0,739 ± 0,025** | **0,643** | **0,803** | **0,768** | **14,5%** |
| U-Net | 0,697 ± 0,025 | 0,578 | 0,779 | 0,728 | 62,5% |

mean ± SD qua seed 0–4. Nguồn: [`11_pixel_tung_seed_6model.csv`](../danh_gia_pixel/11_pixel_tung_seed_6model.csv), [`12_pixel_thong_ke_6model.csv`](../danh_gia_pixel/12_pixel_thong_ke_6model.csv).

**So ghép cặp theo seed (Dice):**

| VM-UNet so với | Δ Dice | paired t p | Wilcoxon p | Số seed VM-UNet cao hơn |
|---|---:|---:|---:|---:|
| Polyp-PVT | −0,155 | 0,0001 | 0,0625 | 0/5 |
| PraNet | −0,146 | 0,0001 | 0,0625 | 0/5 |
| TSVM | −0,115 | 0,0011 | 0,0625 | 0/5 |
| YOLO26s-seg | −0,115 | 0,0009 | 0,0625 | 0/5 |
| U-Net | +0,042 | 0,0166 | 0,0625 | 5/5 |

Với 5 seed, Wilcoxon thấp nhất chỉ được 0,0625. Khoảng cách với 4 mô hình đầu lớn (0,11–0,16) và nhất quán 5/5 seed.

**Xem thêm:**
- Số ảnh polyp có Dice < 0,5 mỗi seed: VM-UNet 12–24 ảnh, Polyp-PVT 2–9 ảnh.
- Theo kích thước polyp (chia 3 nhóm theo diện tích): VM-UNet 0,70 (nhỏ), 0,75 (vừa), 0,77 (lớn); Polyp-PVT 0,88 / 0,90 / 0,90. VM-UNet yếu hơn ở cả ba nhóm, rõ nhất với polyp nhỏ.

## 3. Chi phí

| | VM-UNet | Polyp-PVT | TSVM |
|---|---:|---:|---:|
| Tham số | 27,4 M | 25,1 M | 12,3 M |
| GFLOPs (640×640) | ≥ 66,1 | 85,0 | ≥ 34,8 |
| Huấn luyện | 14,4 giờ/seed | 13,9 giờ/seed | 4,6 giờ/seed |
| VRAM huấn luyện | 12,1 GB (batch 4) | 12,8 GB (batch 8) | |
| Checkpoint | 105 MiB | 96 MiB | 47 MiB |

- FLOPs đếm bằng `FlopCounterMode`, phép quét Mamba không được đếm, nên là cận dưới (giống TSVM).
- **Độ trễ chưa đo.** Trên laptop không có kernel CUDA của `mamba_ssm`, nên đo bằng bản thuần PyTorch sẽ không công bằng. Nếu cần, phải đo trên Kaggle T4, và khi đó không so trực tiếp được với số đo laptop của 5 mô hình kia.

## 4. Diễn giải (dùng khi trả lời hội đồng)

- Ở cùng giao thức với các mô hình khác (640, 100 epoch, batch tương đương 8, tăng cường giống YOLO), **VM-UNet thấp hơn rõ YOLO26s-seg, TSVM, PraNet và Polyp-PVT**, chỉ cao hơn U-Net. Nó cũng báo nhầm 14,5% ảnh nền.
- **Nguyên nhân chưa xác minh.** Các khả năng (giả thuyết):
  1. Cấu hình gốc (Cosine `T_max = 50`) thiết kế cho 300 epoch. Ở 100 epoch, tốc độ học tăng lại trong 50 epoch cuối, làm huấn luyện dao động và seed 0 phân kỳ.
  2. Bản gốc chạy ảnh 256 (ISIC). Ở 640, mỗi lần quét dài 25.600 token, khác xa lúc tiền huấn luyện.
  3. 100 epoch có thể chưa đủ: 3/5 seed đạt tốt nhất ở epoch 97–98.
- **Không kết luận "Mamba kém hơn Transformer".** Đây là một mô hình Mamba cụ thể, cấu hình gốc chưa tinh chỉnh cho bài toán này.
- **Câu nói gợi ý:** "Nhóm đã thử VM-UNet, mô hình Mamba thuần dùng mã gốc của tác giả, ở cùng giao thức. Với cấu hình gốc, nó đạt Dice 0,74, thấp hơn cả YOLO26s-seg. Điều này cho thấy đưa Mamba vào phân đoạn polyp không tự động cho kết quả tốt. Hướng của khóa luận là gắn Mamba vào một khung đã mạnh (YOLO26s-seg) qua TSVM, giữ được Dice 0,85 với chi phí thêm nhỏ."

## 5. Hình

Riêng VM-UNet, ở `hinh/`:

| File | Nội dung |
|---|---|
| `v01_duong_cong_vmunet` | Val Dice và loss theo epoch của 5 seed, kèm lịch tốc độ học |
| `v02_val_dice_tot_nhat` | Val Dice tốt nhất và epoch đạt được của từng seed |

Hình gộp 6 mô hình: xem [`../so_sanh_6_mo_hinh/KET_QUA.md`](../so_sanh_6_mo_hinh/KET_QUA.md).

## 6. Giới hạn

- Val Dice dùng để chọn checkpoint tính trên chính tập val dùng để báo cáo, giống 5 mô hình kia.
- Batch 4 × cộng dồn 2 chỉ xấp xỉ batch 8 (drop_path lấy ngẫu nhiên theo từng mini-batch).
- Không tinh chỉnh siêu tham số cho VM-UNet. Đổi lịch tốc độ học (ví dụ `T_max = 100`) có thể cho kết quả khác, nhưng phải chạy lại 5 seed (khoảng 72 giờ GPU).
