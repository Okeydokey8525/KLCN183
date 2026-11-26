# Polyp-PVT 640: đánh giá theo pixel (seed 0–4)

Chạy ngày 16/11/2026 bằng [`danh_gia/danh_gia_pixel_pvt.py`](../../danh_gia/danh_gia_pixel_pvt.py) trên RTX 5050 Laptop (torch 2.13.0+cu130, timm 1.0.31), suy luận hết 0,7 phút. Log: [`_log_pixel_polyppvt.txt`](_log_pixel_polyppvt.txt).

Kết quả **chỉ dùng cho slide và phần trả lời hội đồng**. Bản báo cáo đã nộp không sửa.

## Giao thức

Giống Bảng 4.8 của báo cáo:
- Nhãn chuẩn vẽ từ polygon YOLO ở độ phân giải gốc.
- Polyp-PVT nhận ảnh kéo về 640×640, chuẩn hóa ImageNet. Lấy `sigmoid(P1 + P2)`, phóng về kích thước gốc, ngưỡng 0,5.
- Chỉ số tính trên từng ảnh của 120 ảnh polyp, lấy trung bình mỗi seed, rồi tính mean ± SD qua seed.
- 40 ảnh nền: tính tỉ lệ ảnh có ít nhất một pixel bị báo là polyp.
- Số của YOLO, TSVM, PraNet, U-Net lấy nguyên từ `Test_Mau/.../pixel_640/10_pixel_tung_anh.csv` và lọc seed 0–4. Không suy luận lại 4 mô hình này.

## Kiểm tra trước khi dùng số

| Kiểm tra | Kết quả |
|---|---|
| Tính lại Val Dice theo đúng cách notebook (ảnh 640, gồm 40 ảnh nền), so với `tom_tat.txt` | seed 0–4 lệch tối đa 0,000007 (do khác GPU T4 và RTX 5050). Checkpoint và mã đã nạp đúng |
| Số theo từng seed của 4 mô hình cũ, so với `11_pixel_tung_seed.csv` gốc | 20/20 dòng khớp, chênh 0 |
| Checkpoint nạp `strict=True` | 5/5 seed |

## Kết quả (mean ± SD, seed 0–4)

| Mô hình | Dice | IoU | Precision | Recall | Ảnh nền báo nhầm (%) |
|---|---:|---:|---:|---:|---:|
| **Polyp-PVT 640** | **0,8937 ± 0,0052** | **0,8308 ± 0,0045** | **0,9089 ± 0,0142** | **0,9125 ± 0,0128** | 3,0 ± 4,1 |
| PraNet 640 | 0,8846 ± 0,0054 | 0,8206 ± 0,0068 | 0,8987 ± 0,0078 | 0,9112 ± 0,0073 | 2,5 ± 0,0 |
| YOLO26s-seg | 0,8543 ± 0,0222 | 0,7979 ± 0,0232 | 0,8671 ± 0,0192 | 0,8728 ± 0,0233 | 0,5 ± 1,1 |
| YOLO26s-seg + TSVM | 0,8538 ± 0,0108 | 0,7998 ± 0,0113 | 0,8575 ± 0,0099 | 0,8772 ± 0,0137 | 2,0 ± 2,1 |
| U-Net 640 | 0,6972 ± 0,0252 | 0,5776 ± 0,0288 | 0,7787 ± 0,0265 | 0,7275 ± 0,0233 | 62,5 ± 3,5 |

Nguồn: [`12_pixel_thong_ke_5model.csv`](12_pixel_thong_ke_5model.csv). Dice khi so với mặt nạ gốc Kvasir-SEG (`raw_dice`) của Polyp-PVT là 0,8943 ± 0,0051, gần như trùng với nhãn polygon.

Ở 5 seed này, TSVM có Dice 0,8538, gần bằng con số 20 seed trong báo cáo (0,8540). Riêng YOLO26s-seg thì 5 seed (0,8543) cao hơn 20 seed (0,8486). Vì vậy **chênh lệch TSVM − YOLO ở đây không đại diện** cho kết luận 20 seed, và không đưa vào slide.

### Dice từng seed

| Seed | Polyp-PVT | PraNet | YOLO26s-seg | TSVM | U-Net |
|---:|---:|---:|---:|---:|---:|
| 0 | 0,8943 | 0,8837 | 0,8727 | 0,8366 | 0,6918 |
| 1 | 0,8848 | 0,8786 | 0,8624 | 0,8659 | 0,6683 |
| 2 | 0,8982 | 0,8858 | 0,8394 | 0,8554 | 0,7169 |
| 3 | 0,8951 | 0,8818 | 0,8232 | 0,8580 | 0,6803 |
| 4 | 0,8960 | 0,8930 | 0,8736 | 0,8529 | 0,7287 |

Polyp-PVT bỏ sót (không dự đoán pixel nào) 1 ảnh polyp ở seed 1, còn 4 seed khác không sót ảnh nào. Ở seed 2, Polyp-PVT báo nhầm 4/40 ảnh nền. Đây là lý do Val Dice của notebook ở seed 2 thấp nhất (0,8986), vì cách tính đó phạt cả ảnh nền, trong khi Dice trên ảnh polyp của seed 2 lại cao nhất (0,8982).

## So ghép cặp cùng seed (n = 5)

Δ = Polyp-PVT − mô hình kia.

| So với | Chỉ số | Δ | Paired t p | Wilcoxon p | Polyp-PVT cao hơn |
|---|---|---:|---:|---:|---:|
| YOLO26s-seg + TSVM | Dice | +0,0399 | 0,003 | 0,0625 | 5/5 |
| | IoU | +0,0309 | 0,008 | 0,0625 | 5/5 |
| | Precision | +0,0514 | 0,004 | 0,0625 | 5/5 |
| | Recall | +0,0353 | 0,004 | 0,0625 | 5/5 |
| | Ảnh nền báo nhầm | +1,0 điểm % | 0,648 | 0,750 | |
| YOLO26s-seg | Dice | +0,0394 | 0,022 | 0,0625 | 5/5 |
| | IoU | +0,0329 | 0,044 | 0,0625 | 5/5 |
| PraNet 640 | Dice | +0,0091 | 0,010 | 0,0625 | 5/5 |
| | IoU | +0,0102 | 0,016 | 0,0625 | 5/5 |
| | Precision | +0,0102 | 0,096 | 0,125 | 4/5 |
| | Recall | +0,0013 | 0,856 | 1,000 | 2/5 |
| | Ảnh nền báo nhầm | +0,5 điểm % | 0,799 | 1,000 | |

Với 5 seed, Wilcoxon hai phía thấp nhất chỉ được p = 0,0625, kể cả khi thắng 5/5. Vì vậy cột Wilcoxon không thể dưới 0,05. Paired t giả định chênh lệch phân phối chuẩn, mà với n = 5 thì không kiểm tra được giả định này.

## Diễn giải (dùng cho slide và phần trả lời)

1. **Transformer (Polyp-PVT) cao nhất ở mọi chỉ số chồng lấn**: Dice 0,894 so với PraNet 0,885, YOLO 0,854, TSVM 0,854. Polyp-PVT cao hơn PraNet ở cả 5/5 seed về Dice và IoU, mức chênh nhỏ (+0,009).
2. **So với TSVM**: Polyp-PVT cao hơn khoảng 0,04 Dice ở cả 5/5 seed. Kết luận này cùng chiều với kết quả PraNet trong báo cáo: các mô hình chuyên phân đoạn ngữ nghĩa, chọn checkpoint theo Dice, cho Dice theo pixel cao hơn mô hình phân đoạn thực thể YOLO.
3. **Báo nhầm ảnh nền**: Polyp-PVT (3,0%), PraNet (2,5%) và TSVM (2,0%) gần nhau, không khác biệt có ý nghĩa. Riêng U-Net báo nhầm 62,5%.
4. **Cách nói đúng**: Polyp-PVT có **xu hướng** cao hơn, thắng 5/5 seed, paired t p < 0,05. Không viết "vượt trội". Không viết "có ý nghĩa thống kê" theo Wilcoxon.

## Giới hạn

- Chỉ có 5 seed, các mô hình trong báo cáo có 20 seed.
- Khác bản gốc của Polyp-PVT: ảnh 640 thay 352, batch 8 thay 16, không huấn luyện đa tỉ lệ, tăng cường giống YOLO. Không so với con số trong bài báo gốc.
- Chọn checkpoint trên chính tập val dùng để báo cáo, giống 4 mô hình còn lại. Không có tập test độc lập.
- Ngưỡng 0,5 cố định, chưa tối ưu.
- Polyp-PVT lớn hơn YOLO26s-seg nhiều và chậm hơn (backbone PVTv2-B2), nhưng chưa đo FLOPs và độ trễ trên cùng máy. Đây là việc có thể làm tiếp.
- Phía Mamba chỉ có TSVM. VM-UNet hết bộ nhớ trên T4 ở 640, batch 8, nên đã bỏ.

## File

| File | Nội dung |
|---|---|
| `10_pixel_tung_anh_polyppvt.csv` | Polyp-PVT × 5 seed × 160 ảnh |
| `11_pixel_tung_seed_5model.csv` | 5 mô hình × seed 0–4 |
| `12_pixel_thong_ke_5model.csv` | mean ± SD và so ghép cặp |
| `du_lieu_dinh_tinh/pred_seed0_polyppvt640.npz` | mặt nạ seed 0 để vẽ ảnh minh họa |
| `*_thu.csv` | lượt chạy thử seed 0 (trùng với seed 0 ở trên) |
