# Hướng 3, bước 0: bộ nào cắt từ video, ảnh nào trùng

Làm ngày 18/11/2026. Mọi bước chỉ đọc `DeepLearning/Dataset` và BG20, không sửa hay di chuyển ảnh gốc nào.

| Bước | Script | Việc làm |
|---|---|---|
| 1 | [`scripts/kiem_tra_video_va_trung.py`](scripts/kiem_tra_video_va_trung.py) | Tính đặc trưng 9.059 ảnh, đo dấu hiệu video của từng bộ |
| 2 | [`scripts/ma_tran_trung_8x8.py`](scripts/ma_tran_trung_8x8.py) | So mọi cặp ảnh của 8 bộ, ghi **mọi cặp** có tương quan ≥ 0,85, có thêm phép lật ngang, lật dọc, xoay 180° |
| 3 | Soi bằng mắt | Mọi cặp **khác bộ** có tương quan ≥ 0,90 (525 cặp). Các cặp khó được phóng to để xem lại |

**Cách đo.**
- md5 của tệp.
- Tương quan Pearson giữa hai ảnh xám thu về 64×64, sau khi cắt viền đen. Mỗi cặp lấy giá trị lớn nhất qua 4 phép: giữ nguyên, lật ngang, lật dọc, xoay 180°.
- Cách đo chép từ bước kiểm tra trùng PolypGen trong Test_Mau.

## 1. Bộ nào cắt từ video

Mốc so sánh là Kvasir-SEG, gồm ảnh riêng lẻ. Trong bộ này, 12,8% ảnh có một ảnh khác cùng bộ giống nó với tương quan ≥ 0,90.

| Bộ | Ảnh | Ảnh có láng giềng ≥ 0,90 trong bộ | Kết luận | Dùng làm |
|---|---:|---:|---|---|
| Kvasir-SEG (BG20) | 1.000 + 200 nền | 12,8% | ảnh riêng lẻ | đã huấn luyện |
| PolypGen ảnh đơn C1–C6 | 1.537 | 16,3% | gần như ảnh riêng lẻ. C4 có 42%: vài khung chụp cùng một polyp | **gộp** |
| PolypDB Simula WLI | 2.563 | 6,2% | ảnh riêng lẻ | **gộp** |
| PolypDB Simula NBI | 141 | 4,3% | ảnh riêng lẻ | **gộp** |
| PolypDB Karolinska | 30 | 13,3% | ảnh riêng lẻ | **gộp** |
| CVC-ClinicDB | 612 | 61,4% | **video** (`metadata.csv` ghi sequence) | test ngoài |
| CVC-ColonDB | 380 | 68,4% | **video**, có 86 cặp trùng md5 ngay trong bộ | test ngoài |
| ETIS-Larib | 196 | 73,5% | **video**, có 3 cặp trùng md5 trong bộ | test ngoài |
| BKAI-IGH, PolypDB/BKAI | 1.200 mỗi bộ | 46–60% | nhiều khung cho mỗi bệnh nhân. Bản Kaggle không có mã bệnh nhân, phần test không có mặt nạ | **không dùng** |

Hình các cặp gần nhất trong từng bộ: [`hinh/h02_cap_gan_nhat_trong_bo.png`](hinh/h02_cap_gan_nhat_trong_bo.png).

## 2. Ma trận trùng 8×8

Mỗi ô ghi số cặp ảnh. Cột "soi mắt" chỉ áp dụng cho các cặp khác bộ có tương quan ≥ 0,90. Nguồn: [`bang/08_ma_tran_8x8.csv`](bang/08_ma_tran_8x8.csv); mọi cặp ≥ 0,85 ở [`bang/07_cap_ge_0_85.csv`](bang/07_cap_ge_0_85.csv).

**Giữa hai bộ khác nhau (28 cặp bộ):**

| Bộ A | Bộ B | md5 | ≥ 0,95 | 0,90–0,95 | 0,85–0,90 | Soi mắt (các cặp ≥ 0,90) |
|---|---|---:|---:|---:|---:|---|
| BG20 | PolypGen | 0 | 0 | 3 | 405 | 3 ảnh khác |
| BG20 | **Simula WLI** | 0 | 25 | 80 | 1.231 | **24 cùng ảnh, 1 cùng tổn thương khác khung**, 80 ảnh khác |
| BG20 | Simula NBI | 0 | 0 | 0 | 38 | |
| BG20 | Karolinska | 0 | 0 | 1 | 67 | 1 ảnh khác |
| BG20 | CVC-ClinicDB | 0 | 0 | 2 | 172 | 2 ảnh khác |
| BG20 | CVC-ColonDB | 0 | 0 | 0 | 48 | |
| BG20 | ETIS-Larib | 0 | 0 | 3 | 164 | 3 ảnh khác |
| PolypGen | Simula WLI | 0 | 0 | 61 | 1.357 | 61 ảnh khác |
| PolypGen | Simula NBI | 0 | 0 | 0 | 33 | |
| PolypGen | Karolinska | 0 | 0 | 9 | 87 | 9 ảnh khác |
| PolypGen | CVC-ClinicDB | 0 | 0 | 55 | 994 | 55 ảnh khác |
| PolypGen | CVC-ColonDB | 0 | 0 | 85 | 903 | 85 ảnh khác |
| PolypGen | ETIS-Larib | 0 | 0 | 51 | 581 | 51 ảnh khác |
| Simula WLI | Simula NBI | 0 | 1 | 6 | 95 | **1 cùng tổn thương** (cùng ngày 29/04/2013, một ảnh WLI và một ảnh NBI), 6 ảnh khác |
| Simula WLI | Karolinska | 0 | 0 | 5 | 94 | 5 ảnh khác |
| Simula WLI | CVC-ClinicDB | 0 | 0 | 4 | 494 | 4 ảnh khác |
| Simula WLI | CVC-ColonDB | 0 | 0 | 55 | 441 | 55 ảnh khác |
| Simula WLI | ETIS-Larib | 0 | 0 | 25 | 398 | 25 ảnh khác |
| Simula NBI | Karolinska | 0 | 0 | 1 | 3 | 1 ảnh khác |
| Simula NBI | CVC-ClinicDB / ColonDB / ETIS | 0 | 0 | 0 | 44 / 18 / 7 | |
| Karolinska | CVC-ClinicDB / ColonDB | 0 | 0 | 0 | 65 / 31 | |
| Karolinska | ETIS-Larib | 0 | 0 | 3 | 69 | 3 ảnh khác |
| CVC-ClinicDB | CVC-ColonDB | 0 | 0 | 30 | 1.029 | 30 ảnh khác |
| CVC-ClinicDB | ETIS-Larib | 0 | 0 | 0 | 237 | |
| CVC-ColonDB | ETIS-Larib | 0 | 0 | 20 | 187 | 20 ảnh khác |

**Trong cùng một bộ (8 ô).** Các cặp này không cần soi bằng mắt. Khi chia, mọi ảnh có tương quan ≥ 0,90 được gom thành một cụm và cả cụm vào cùng một tập.

| Bộ | md5 | ≥ 0,95 | 0,90–0,95 |
|---|---:|---:|---:|
| BG20 | 0 | 7 | 109 |
| PolypGen | 2 | 30 | 216 |
| Simula WLI | 0 | 10 | 139 |
| Simula NBI | 0 | 0 | 1 |
| Karolinska | 0 | 0 | 2 |
| CVC-ClinicDB | 0 | 597 | 1.151 |
| CVC-ColonDB | 86 | 425 | 879 |
| ETIS-Larib | 3 | 118 | 112 |

## 3. Kết quả soi bằng mắt

Đã soi đủ **525 cặp khác bộ** có tương quan ≥ 0,90. Hình soi nằm ở `hinh/soi_8x8/soi_01..14.png`; 9 cặp khó được phóng to ở `soi_phong_to.png`. Kết luận từng cặp: [`bang/09_cap_khac_bo_ge_0_90_can_soi.csv`](bang/09_cap_khac_bo_ge_0_90_can_soi.csv).

- **Trùng thật chỉ có ở Simula WLI với BG20:**
  - 24 cặp là cùng một ảnh, tương quan 0,959–0,993;
  - 1 cặp (#46, tương quan 0,920) là cùng tổn thương, cùng thời điểm chụp 12/09/2012 00:18:32.
- **Simula WLI với Simula NBI:** 1 cặp là cùng tổn thương chụp hai chế độ. Hai ảnh này phải vào cùng một tập khi chia.
- **Các cặp còn lại là ảnh khác**, kể cả cặp 0,952. Ảnh nội soi cùng tông màu dễ cho tương quan 0,90–0,95 dù là hai ảnh khác nhau.
- **Không có ảnh nào của bộ test ngoài (CVC, ETIS) trùng với các bộ sẽ gộp.**
- **Sửa lại so với lần trước.** Lần trước mình báo 27 ảnh cần loại, gồm 3 cặp "cùng tổn thương". Phóng to ra thì 2 trong 3 cặp đó (tương quan 0,932 và 0,903) khác ngày chụp và khác polyp, nên không phải trùng. Số đúng là **25 ảnh**.

**Thư mục [`trung_lap/`](trung_lap/)** chứa **bản chép** của 26 cặp trùng thật, mỗi cặp một thư mục con. Kèm theo là `danh_sach_trung.csv`. Ảnh gốc không bị đụng đến.

**Sửa lại lần 2 (19/11).** Khi chia dữ liệu, mình dùng thêm khớp điểm đặc trưng ORB + RANSAC và tìm ra **37 ảnh Simula WLI nữa là bản sao ảnh Kvasir-SEG**. Bản Simula còn hộp xanh ở góc dưới trái, còn bản Kvasir đã bôi đen hộp này. Vì vậy tương quan ảnh xám chỉ đạt 0,65–0,88 và phép đo ở trên bỏ sót. Đã soi bằng mắt cả 37 cặp: đều là cùng một ảnh (hình `../chon_du_lieu/hinh/h06_trung_khac_bo_khop_diem.jpg`). Tổng số ảnh Simula WLI trùng Kvasir là **62** (25 + 37). Các ảnh này bị loại ngay ở bước chọn, xem [`../chon_du_lieu/trung_khac_bo_khop_diem.csv`](../chon_du_lieu/trung_khac_bo_khop_diem.csv). Hệ quả: Simula WLI chỉ còn khoảng 2.496 ảnh dùng được. Phần giới hạn ở mục 5 ("ảnh bị cắt mạnh ... có thể không bị bắt") đã xảy ra thật.

## 4. Còn lại để gộp

| Nguồn | Ảnh dùng được | Ghi chú |
|---|---:|---|
| Kvasir-SEG (BG20 cũ) | 1.000 + 200 nền | |
| PolypGen ảnh đơn C1–C6 | 1.537 | 1.412 ảnh có polyp, 125 ảnh mặt nạ rỗng |
| PolypDB Simula WLI | **2.538** | 2.563 trừ 25 ảnh trùng BG20 ([`bang/06_anh_loai_vi_trung_BG20.csv`](bang/06_anh_loai_vi_trung_BG20.csv)) |
| PolypDB Simula NBI | 141 | |
| PolypDB Karolinska | 30 | |

Test ngoài vẫn là CVC-ClinicDB (612 ảnh), CVC-ColonDB (380) và ETIS-Larib (196), như trong báo cáo.

## 5. Giới hạn của cách kiểm

- Ảnh bị cắt mạnh, phóng to hoặc xoay góc lẻ có thể không bị bắt. Phép đo chỉ thử lật và xoay 180°.
- Dải 0,85–0,90 chưa soi bằng mắt: có 9.292 cặp khác bộ trong dải này. Trong dải 0,90–0,95, chỉ 1/499 cặp khác bộ là trùng thật, nên khả năng sót ở dải thấp hơn là nhỏ, nhưng chưa được kiểm chứng.
- Chưa kiểm tra mặt nạ. Việc này để đến bước đóng gói.

## Bảng và hình cũ

`bang/05_soi_tay_bg20.csv` và `hinh/soi_bg20_*.png`, `hinh/soi_giua_bo.png` là của lần soi đầu, đã được bảng `09_*` và `hinh/soi_8x8/` thay thế. Giữ lại để đối chiếu.
