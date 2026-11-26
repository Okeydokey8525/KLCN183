# Polyp-PVT là gì? (giải thích ngắn gọn)

## 1. Tóm tắt một câu

**Polyp-PVT** là mô hình **phân đoạn ngữ nghĩa** (semantic segmentation) chuyên cho ảnh nội soi đại tràng. Với mỗi điểm ảnh, mô hình cho biết điểm đó thuộc polyp hay không.

Phần trích đặc trưng (backbone) là **PVTv2-B2**, một Vision Transformer dạng kim tự tháp. Mô hình do Bo Dong và cộng sự công bố năm 2021, bài báo *"Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformers"*, đăng trên tạp chí CAAI Artificial Intelligence Research năm 2023. Mã nguồn công khai trên GitHub (DengPingFan/Polyp-PVT).

## 2. Tại sao cần Polyp-PVT?

Các mô hình CNN trước đó (như PraNet, dùng ResNet-50) nhìn ảnh qua các ô cửa sổ nhỏ. Vì vậy chúng dễ gặp khó khi:
- polyp có màu gần giống niêm mạc xung quanh (polyp "ngụy trang");
- polyp quá nhỏ hoặc quá lớn;
- ảnh bị lóa sáng hoặc nhòe do chuyển động.

Transformer dùng cơ chế **self-attention**: mỗi vùng ảnh "nhìn" được mọi vùng khác, nên mô hình hiểu được ngữ cảnh của toàn bộ ảnh. Polyp-PVT tận dụng điểm mạnh này, rồi thêm ba khối xử lý riêng để giữ lại chi tiết nhỏ.

## 3. Kiến trúc (sơ đồ đơn giản)

```
Ảnh nội soi
   │
   ▼
PVTv2-B2 (Transformer kim tự tháp) ──► 4 tầng đặc trưng X1 (chi tiết) … X4 (ngữ nghĩa)
   │
   ├─ X2, X3, X4 ──► CFM  (Cascaded Fusion Module): gộp đặc trưng cấp cao → vị trí, hình dạng thô của polyp
   ├─ X1        ──► CIM  (Camouflage Identification Module): lọc nhiễu, tìm chi tiết/biên polyp "ngụy trang"
   └─ CFM + CIM ──► SAM  (Similarity Aggregation Module): lan thông tin vị trí sang chi tiết, dựa trên độ tương đồng
   │
   ▼
Mặt nạ polyp (mỗi pixel: polyp / không phải polyp)
```

Có thể hình dung như bác sĩ đọc ảnh:
- **CFM** giống việc nhìn tổng thể để biết polyp nằm khoảng chỗ nào.
- **CIM** giống việc soi kỹ phần biên để thấy chỗ polyp lẫn vào niêm mạc.
- **SAM** ghép hai nguồn thông tin trên để vẽ ra đường viền cuối cùng.

Hàm mất mát là weighted BCE cộng weighted IoU, cùng kiểu với PraNet: những điểm ảnh gần biên được tính trọng số lớn hơn.

## 4. Polyp-PVT trong khóa luận của nhóm (Hướng 1, dùng làm mô hình so sánh)

Nhóm huấn luyện lại Polyp-PVT trên cùng dữ liệu với mô hình chính:
- Kvasir-SEG có thêm 20% ảnh nền, ảnh 640, 5 seed;
- huấn luyện trên Kaggle Tesla T4;
- đánh giá theo pixel giống nhau cho cả 6 mô hình.

| Mô hình | Kiểu | Dice pixel (5 seed) | Tham số | GFLOPs | Giờ huấn luyện mỗi seed | Ảnh nền bị báo nhầm |
|---|---|---:|---:|---:|---:|---:|
| **Polyp-PVT** | Transformer | **0,894 ± 0,005** | 25,1 M | 85,0 | 13,9 | 3,0% |
| PraNet | CNN | 0,885 ± 0,005 | 24,6 M | 72,4 | 4,0 | 2,5% |
| YOLO26s-seg | YOLO | 0,854 ± 0,022 | 11,4 M | 34,3 | 2,3 | 0,5% |
| YOLO26s-seg + TSVM (mô hình của nhóm) | YOLO + Mamba | 0,854 ± 0,011 | 12,3 M | ≥ 34,8 | 4,6 | 2,0% |
| VM-UNet | Mamba | 0,739 ± 0,025 | 27,4 M | ≥ 66,1 | 14,4 | 14,5% |
| U-Net | CNN | 0,697 ± 0,025 | 7,8 M | 150,9 | 5,7 | 62,5% |

Đọc bảng này:
- **Polyp-PVT có Dice pixel cao nhất** trong 6 mô hình. Tuy vậy, nó chỉ hơn PraNet khoảng 0,009, mà mới có 5 seed. Vì thế chỉ nên nói là *có xu hướng cao hơn*, chưa đủ để khẳng định chắc chắn.
- **Đổi lại, chi phí rất lớn:**
  - huấn luyện 13,9 giờ mỗi seed, gấp khoảng 6 lần YOLO;
  - cần tới 12,8 GB VRAM;
  - trên laptop RTX 5050, Polyp-PVT chậm hơn YOLO khoảng 2,7–3,1 lần.
- **Khác biệt về loại đầu ra:**
  - Polyp-PVT chỉ cho **một mặt nạ chung**. Nếu ảnh có 2 polyp, nó không tách ra được đâu là polyp thứ nhất, đâu là polyp thứ hai.
  - YOLO và TSVM làm **phân đoạn từng đối tượng** (instance segmentation): mỗi polyp có hộp bao, mặt nạ và độ tin cậy riêng. Nhờ vậy đếm được số polyp và dễ dùng cho video theo thời gian thực.

## 5. Tóm tắt nhanh (Pareto 80/20)

- Polyp-PVT = Transformer PVTv2 cộng ba khối CFM, CIM, SAM, chuyên phân đoạn polyp theo pixel.
- Điểm mạnh: hiểu ngữ cảnh toàn ảnh, biên polyp chính xác; Dice cao nhất trong thí nghiệm của nhóm.
- Điểm yếu: nặng và chậm, tốn nhiều giờ huấn luyện và nhiều VRAM, không tách riêng từng polyp.

## 6. Ba câu tự kiểm tra

1. Vì sao Transformer xử lý polyp "ngụy trang" tốt hơn CNN thuần?
2. CFM và CIM khác nhau thế nào: mỗi khối dùng đặc trưng tầng nào, và để làm gì?
3. Vì sao Dice của Polyp-PVT cao hơn mà nhóm vẫn chọn YOLO26s-seg + TSVM làm mô hình chính?

*Gợi ý câu 3: cần tách từng polyp, cần tốc độ cho video, và chi phí huấn luyện thấp hơn khoảng 3 lần.*
