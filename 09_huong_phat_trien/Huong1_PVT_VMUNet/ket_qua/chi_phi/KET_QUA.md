# Chi phí của 5 mô hình: thời gian huấn luyện, tham số, FLOPs, độ trễ

Chạy ngày 16/11/2026 bằng [`danh_gia/do_chi_phi_5_mo_hinh.py`](../../danh_gia/do_chi_phi_5_mo_hinh.py). Log: [`_log_chi_phi.txt`](_log_chi_phi.txt). Kết quả chỉ dùng cho slide và phần trả lời hội đồng.

## Cách lấy số

| Đại lượng | Nguồn | Ghi chú |
|---|---|---|
| Thời gian huấn luyện | Dòng cuối `results.csv` (giây cộng dồn) của seed 0–4 | Gồm cả thời gian val mỗi epoch. 100 epoch, batch 8, ảnh 640 cho cả 5 mô hình. Polyp-PVT chạy trên Tesla T4 (ghi trong `tom_tat.txt`). Với 4 mô hình kia, log có sẵn không ghi loại GPU, chỉ biết là Kaggle |
| Tham số, FLOPs | Checkpoint seed 0, ảnh 1×3×640×640 | FLOPs đếm bằng `FlopCounterMode` (conv, matmul). Phép theo phần tử không được đếm, nên số của TSVM là **cận dưới** |
| VRAM suy luận | `max_memory_allocated` sau 1 lần forward, batch 1 | Chỉ phần PyTorch cấp phát, không gồm bộ nhớ CUDA context |
| Độ trễ GPU | RTX 5050 Laptop, FP32, batch 1, chỉ forward, 5 vòng × 200 lần đo, đảo thứ tự mỗi vòng | Cách đo chép từ `do_flops_gpu.py` của báo cáo. Đo lại cả 5 mô hình trong **cùng một phiên** |

## Kết quả

| Mô hình | Kiểu | Dice pixel (seed 0–4) | Tham số (triệu) | GFLOPs | Huấn luyện (giờ/seed) | Độ trễ p10 (ms) | Tỉ lệ độ trễ so với YOLO | VRAM suy luận (MB) | Checkpoint (MiB) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLO26s-seg | YOLO, phân đoạn thực thể | 0,854 | 11,4 | 34,3 | 2,3 ± 0,1 | 12,1 | 1,00 | 142 | 22 (FP16) |
| YOLO26s-seg + TSVM | YOLO + Mamba | 0,854 | 12,3 | ≥ 34,8 | 4,6 ± 0,1 | 21,1 | 1,75 | 199 | 47 |
| PraNet (ResNet-50) | CNN | 0,885 | 24,6 | 72,4 | 4,0 ± 0,1 | 15,5 | 1,29 | 297 | 94 |
| U-Net | CNN | 0,697 | 7,8 | 150,9 | 5,7 ± 0,8 | 23,9 | 1,98 | 694 | 30 |
| **Polyp-PVT (PVTv2-B2)** | Transformer | **0,894** | **25,1** | **85,0** | **13,9 ± 0,1** | **34,6** | **2,87** | **512** | **96** |

Nguồn: [`20_thoi_gian_huan_luyen_tung_seed.csv`](20_thoi_gian_huan_luyen_tung_seed.csv), [`21_flops_do_tre_gpu_5model.csv`](21_flops_do_tre_gpu_5model.csv). Dice lấy từ [`../danh_gia_pixel/KET_QUA.md`](../danh_gia_pixel/KET_QUA.md).

Polyp-PVT dùng tối đa **12,8 GB** VRAM khi huấn luyện (batch 8, T4 16 GB). Bốn mô hình kia không ghi số này.

### Độ trễ: phải đọc thận trọng

Trung vị từng vòng (ms):

| Vòng | YOLO | TSVM | PraNet | U-Net | Polyp-PVT |
|---:|---:|---:|---:|---:|---:|
| 0 | 39,4 | 48,6 | 39,8 | 24,2 | 42,2 |
| 1 | 30,3 | 54,1 | 40,0 | 24,0 | 40,3 |
| 2 | 31,3 | 39,3 | 23,6 | 24,0 | 36,8 |
| 3 | 12,9 | 22,0 | 15,9 | 24,1 | 35,1 |
| 4 | 13,2 | 22,0 | 15,7 | 24,2 | 35,0 |

- Ba vòng đầu, các mô hình nhỏ (YOLO, TSVM, PraNet) chậm gấp 2–3 lần hai vòng cuối. U-Net và Polyp-PVT hầu như không đổi. Nhiệt độ GPU luôn dưới 55 °C trước mỗi khối và xung nhịp lên 2.805–2.842 MHz khi chạy, nên nguyên nhân không phải GPU nóng.
- Giả thuyết **chưa xác minh**: mô hình nhỏ bị giới hạn bởi phía CPU (gọi kernel, chế độ nguồn), còn mô hình nặng bị giới hạn bởi GPU. Điều này giống ghi chú ở `09_HIEU_NANG_TINH_TOAN.md` của báo cáo.
- Vì vậy bảng dùng **phân vị 10%**, giá trị ít bị ảnh hưởng nhất. Theo giá trị nhỏ nhất, tỉ lệ là: TSVM 1,80, PraNet 1,35, U-Net 2,18, Polyp-PVT 3,13. Hai vòng cuối cho kết quả tương tự (TSVM 1,67–1,70, Polyp-PVT 2,65–2,71).
- Tỉ lệ TSVM/YOLO ở hai vòng ổn định (1,67–1,70) gần với lần đo 1 trong báo cáo (1,73). Báo cáo dùng lần đo 2 (1,51). **Chỉ nói "TSVM chậm hơn YOLO khoảng 1,5–1,8 lần"**, không đưa một con số ms cố định.
- Đây là kết quả trên một laptop, một checkpoint seed 0. Không suy ra tốc độ trên máy nội soi thật.

## Ưu điểm và hạn chế (dùng khi trả lời hội đồng)

| Mô hình | Ưu điểm | Hạn chế |
|---|---|---|
| YOLO26s-seg | Nhẹ nhất (11,4 M, 34 GFLOPs), nhanh nhất, huấn luyện 2,3 giờ. Cho cả hộp và mặt nạ từng polyp, đếm được số polyp | Dice pixel thấp hơn các mô hình phân đoạn ngữ nghĩa khoảng 0,03–0,04 |
| YOLO26s-seg + TSVM | Chỉ thêm 7% tham số và khoảng 1,5% FLOPs. Giữ được ưu điểm phân đoạn thực thể của YOLO | Chậm hơn YOLO khoảng 1,5–1,8 lần, vì phép quét SS2D viết bằng PyTorch thuần. Huấn luyện gấp đôi thời gian. Dice pixel ngang YOLO (khác biệt ở 20 seed xem báo cáo) |
| PraNet | Dice 0,885, đứng thứ hai. Độ trễ chỉ khoảng 1,3 lần YOLO | Tham số gấp đôi YOLO, FLOPs gấp đôi. Chỉ cho mặt nạ, không tách từng polyp |
| U-Net | Ít tham số nhất (7,8 M) | FLOPs cao nhất (151 G) vì tích chập ở độ phân giải đầy đủ. Dice thấp nhất (0,697), báo nhầm 62,5% ảnh nền |
| Polyp-PVT | **Dice cao nhất (0,894)**, thắng PraNet 5/5 seed | **Nặng nhất**: 25,1 M tham số, 85 GFLOPs, chậm khoảng 2,9–3,1 lần YOLO. Huấn luyện **13,9 giờ/seed**, gấp 6 lần YOLO và 3 lần TSVM. Cần 12,8 GB VRAM để huấn luyện batch 8. Chỉ cho mặt nạ |

**Câu nói gợi ý:** Polyp-PVT cho Dice cao nhất, nhưng đánh đổi bằng chi phí lớn nhất. Nó có gấp đôi tham số, gấp 2,5 lần FLOPs, chậm khoảng 3 lần và tốn gấp 6 lần thời gian huấn luyện so với YOLO26s-seg. Khóa luận chọn hướng YOLO vì cần một mô hình nhẹ, cho được từng polyp riêng. TSVM là thử nghiệm gắn Mamba vào khung nhẹ đó, không cạnh tranh trực tiếp với mô hình chuyên phân đoạn ngữ nghĩa.

## Hình

Vẽ bằng [`danh_gia/ve_hinh_chi_phi.py`](../../danh_gia/ve_hinh_chi_phi.py), lưu ở `hinh/` (PNG 300 dpi và PDF):

| File | Nội dung |
|---|---|
| `c01_chi_phi_5_mo_hinh` | 6 ô: Dice, tham số, GFLOPs, giờ huấn luyện, tỉ lệ độ trễ, VRAM suy luận |
| `c02_dice_va_do_tre` | Dice theo tỉ lệ độ trễ, cỡ chấm theo số tham số (đánh đổi độ chính xác và chi phí) |
| `c03_thoi_gian_huan_luyen` | Giờ huấn luyện mỗi seed, có chấm từng seed và hệ số so với YOLO |

## Giới hạn

- Thời gian huấn luyện lấy từ log Kaggle. Chỉ Polyp-PVT có ghi loại GPU (T4). Nếu các lượt khác chạy trên GPU khác thì so sánh lệch.
- Độ trễ không ổn định giữa các vòng ở mô hình nhỏ (xem trên). Có thể đo lại thêm một phiên, đặt tên `_lan2`, giữ nguyên phiên này.
- FLOPs của TSVM là cận dưới. `FlopCounterMode` không đếm phép quét theo phần tử.
