# Hướng 3: gộp dữ liệu rồi chia 3 tập, YOLO26s-seg và TSVM

Chốt ngày 18/11/2026: chạy trước **5 seed cho mỗi mô hình**, tổng 10 lượt. Nếu còn thời gian thì nâng lên 10 seed cho mỗi mô hình. Siêu tham số giữ như 40 lượt trong báo cáo. Kết quả chỉ dùng cho slide và phần trả lời hội đồng.

## Tình trạng

| Bước | Tình trạng |
|---|---|
| 0. Kiểm tra dữ liệu: bộ nào cắt từ video, ma trận trùng 8×8, soi bằng mắt | **Xong.** Xem [`kiem_tra_du_lieu/KET_QUA.md`](kiem_tra_du_lieu/KET_QUA.md) |
| 1a. Gom mọi ảnh ứng viên vào [`anh_gom/`](anh_gom/), thu nhỏ về cạnh dài 640 | **Xong 19/11.** Người dùng soi ảnh gần giống bằng ứng dụng riêng |
| 1b. Chốt số ảnh gộp, tỉ lệ ảnh nền, tỉ lệ chia 3 tập | Đã chốt khoảng 3.000 ảnh polyp, ảnh âm tính 10–15% (tổng 3.333–3.529, train khoảng 2.400), dừng an toàn ở 11,8 giờ. Chờ chọn ảnh cụ thể |
| 2. Chọn 3.000 polyp + 509 âm tính (14,5%), kiểm tra mặt nạ, gom cụm, chia 70/15/15 | **Xong 19/11, chờ người dùng duyệt hình.** [`scripts/chon_va_chia.py`](scripts/chon_va_chia.py), kết quả ở [`chon_du_lieu/`](chon_du_lieu/): `danh_sach_chon.csv`, `tom_tat_chia.csv`, `hinh/h01–h05`. Train 2.455, val 532, test 522. Lần 2 (theo duyệt 19/11): bỏ 14 ảnh polyp có mặt nạ < 0,2% hoặc > 70% ảnh (`anh_bo_mat_na_bat_thuong.csv`), Simula WLI bù lên 425 ảnh. Lần 4 (19/11): thêm khớp điểm ORB + RANSAC ([`scripts/khop_diem_dac_trung.py`](scripts/khop_diem_dac_trung.py)) để (a) gom khung liền nhau của cùng video vào một tập, vd. C2 001160/001161 mà người dùng phát hiện (tương quan chỉ 0,86); (b) loại thêm **37 ảnh Simula WLI là bản sao ảnh Kvasir** mà phép tương quan bỏ sót (bản Simula còn hộp xanh, bản Kvasir bị bôi đen), xem `chon_du_lieu/trung_khac_bo_khop_diem.csv` và `hinh/h06_*`. Không cặp cùng cảnh nào nằm ở hai tập khác nhau. Train 2.455, val 532, test 522. Các lần trước giữ ở `chon_du_lieu_lan1_*`, `_lan2_*`, `_lan3_*` |
| 2b. Tạo bộ YOLO 3 tập [`du_lieu/H3_Gop_3tap/`](du_lieu/H3_Gop_3tap/) (images, labels, masks, `data_h3_3tap.yaml`) | **Xong 19/11** bằng [`scripts/dong_goi_3tap.py`](scripts/dong_goi_3tap.py), cách đổi mặt nạ sang polygon chép từ script BG20 của báo cáo. Kiểm tra: 3.000/3.000 ảnh polyp có polygon, 509/509 ảnh âm tính nhãn rỗng, Dice polygon so với mặt nạ trung vị 0,995 (thấp nhất 0,961). Hình mẫu `chon_du_lieu/hinh/h08_*` |
| 2b'. Đổi tên 7 tệp có ký tự Kaggle không nhận (`]`, `(`, `)`): `00489].jpg` → `00489.jpg`, `SN_..._(NBI)` → `SN_..._NBI` | [`scripts/doi_ten_an_toan_kaggle.py`](scripts/doi_ten_an_toan_kaggle.py); bảng đối chiếu `du_lieu/H3_Gop_3tap/doi_ten_kaggle.csv` |
| 2c. Gói Kaggle [`Goi_Kaggle_H3/`](Goi_Kaggle_H3/): zip 255 MB (không kèm masks), 8 notebook chạy thật (YOLO s0; s1+s2; s3+s4; TSVM mỗi seed 1 phiên), 1 notebook chạy thử, 1 notebook chạy tiếp | **Xong 19/11** bằng [`scripts/tao_goi_kaggle_h3.py`](scripts/tao_goi_kaggle_h3.py). Mốc an toàn 11,8 giờ dừng cuối epoch và giữ `last_resume.pt`; đã chạy thử dừng + chạy tiếp trên máy (tập con, YOLO26n): dừng sau epoch 1, chạy tiếp epoch 2–3, `results.csv` liền mạch. Hướng dẫn: [`Goi_Kaggle_H3/HUONG_DAN.md`](Goi_Kaggle_H3/HUONG_DAN.md) |
| 3. Huấn luyện trên Kaggle | Người dùng chạy |
| 4. Đánh giá test + test ngoài (CVC, ETIS), thống kê | Chưa làm |

## Dữ liệu (sau bước 0)

- **Đường dẫn PolypDB:** `DeepLearning/Dataset/PolypDB/PolypDB_center_wise/{Simula/WLI, Simula/NBI, Karolinska/WLI}` (cùng ảnh có trong `PolypDB_modality_wise/{WLI,NBI}`, script đọc từ đây). Tiền tố trong bộ dữ liệu: `SW_`, `SN_`, `KA_`.
- **Gộp:**
  - Kvasir-SEG (BG20 cũ);
  - PolypGen ảnh đơn (1.537 ảnh);
  - PolypDB Simula WLI (2.538 ảnh, đã bỏ 25 ảnh trùng với Kvasir);
  - PolypDB Simula NBI (141 ảnh);
  - PolypDB Karolinska (30 ảnh).
- **Test ngoài** (cắt từ video, không gộp): CVC-ClinicDB, CVC-ColonDB, ETIS-Larib.
- **Không dùng:**
  - BKAI-IGH, PolypDB/BKAI: nhiều khung cho mỗi bệnh nhân;
  - PolypGen `sequenceData`: là video.

## Kiểm tra chéo bằng Krokiet (19/11)

Người dùng quét `anh_gom/` (polyp + âm tính, 10.521 ảnh) bằng Krokiet 12.0.2 (Czkawka, *Similar Images*, Hash Size 16, Gradient, Max difference 10/40), lưu ở `anh_gom/anh_trung/`. Đối chiếu bằng [`scripts/doi_chieu_krokiet.py`](scripts/doi_chieu_krokiet.py), kết quả ở `chon_du_lieu/doi_chieu_krokiet.csv` và `hinh/h07b_krokiet_khac_nguon.jpg`.

- 824 nhóm, 2.077 ảnh. 780 nhóm là các khung trong chuỗi video âm tính (đúng như dự kiến).
- **0 nhóm có ảnh đã chọn nằm ở hai tập khác nhau.** Có 1 nhóm (PG_C2 001140/001346) script để ở hai cụm khác nhau, nhưng cả hai ảnh đều ở train.
- 5 nhóm khác nguồn: 4 cặp ảnh NBI/WLI trùng tên (đã biết; chỉ giữ bản NBI) và 1 ảnh Simula WLI `02bb84bf-1a2e-476f…` thật ra là ảnh Kvasir normal-cecum (trùng tên, trùng nội dung). Ảnh này không có trong bộ đã chọn.
- Lần quét thứ hai trên bộ 3 tập `du_lieu/H3_Gop_3tap/` (kết quả chuyển sang `chon_du_lieu/krokiet_quet_H3_Gop_3tap/`): 15 nhóm, 33 ảnh, **cả 15 nhóm đều nằm trong train**.
- Krokiet không bắt được 37 cặp Kvasir/Simula có hộp xanh mà khớp điểm ORB đã tìm ra, nên hai cách kiểm tra bổ sung cho nhau.

## Thư mục ảnh gom (`anh_gom/`)

Tạo bằng [`scripts/gom_anh_640.py`](scripts/gom_anh_640.py). Đây là bản sao: ảnh có cạnh dài > 640 được thu nhỏ về 640 (giữ tỉ lệ, LANCZOS, JPEG chất lượng 95); ảnh nhỏ hơn giữ nguyên, không phóng to. Ảnh gốc không bị đụng. Bảng [`anh_gom/danh_sach_anh.csv`](anh_gom/danh_sach_anh.csv) nối tên mới với đường dẫn gốc, kích thước cũ/mới và md5 gốc.

| Thư mục | Tiền tố | Nguồn | Ảnh |
|---|---|---|---:|
| `polyp/` | `KV_` | Kvasir-SEG (BG20 cũ) | 1.000 |
| | `PG_C1_` … `PG_C6_` | PolypGen ảnh đơn có polyp | 1.412 |
| | `SW_` | PolypDB Simula WLI (đã bỏ 25 ảnh trùng BG20) | 2.533 |
| | `SN_` | PolypDB Simula NBI | 146 |
| | `KA_` | PolypDB Karolinska | 30 |
| `am_tinh/` | `CE_` | Kvasir normal-cecum (200 ảnh đã dùng trong BG20, cột `da_dung_BG20`) | 1.000 |
| | `PGR_C?_` | PolypGen ảnh đơn mặt nạ rỗng | 125 |
| | `PGS_seqNN_` | PolypGen chuỗi âm tính, mọi khung của 23 chuỗi | 4.275 |
| | | **Tổng** | **10.521** (5.121 polyp, 5.400 âm tính) |

- Cạnh dài sau khi thu nhỏ: 9.527 ảnh đúng 640. 994 ảnh nhỏ hơn vì ảnh gốc đã nhỏ: 953 ảnh Kvasir (trung vị 621, nhỏ nhất 415), 26 ảnh PolypGen C5 (nhỏ nhất 384), 15 ảnh PolypGen C1.
- Có 2 cặp trùng md5 ngay trong PolypGen C5: `50000140`/`50000141` và `50000284`/`50000285`. Khi chia, mỗi cặp phải vào cùng một tập (hoặc bỏ một ảnh).
- **Sửa số so với bước 0.** Bảng `01_tung_anh.csv` xếp 5 ảnh trong thư mục NBI vào nhóm Simula/WLI. 5 ảnh này trùng tên với 5 ảnh WLI nhưng khác nội dung. Số đúng là: Simula WLI **2.558** ảnh (dùng được 2.533 sau khi bỏ 25 ảnh trùng), Simula NBI **146** ảnh. Năm cặp trùng tên (`100H0004`, `25e89b9e…`, `5300b629…`, `7c5c7bee…`, `852816d5…`) nhiều khả năng là cùng tổn thương chụp hai chế độ, nên khi chia phải cho mỗi cặp vào cùng một tập.
- Chuỗi âm tính có **4.275 khung** ảnh .jpg, không phải 4.325 như đã báo trước đó.

## Thư mục

```
Huong3_GopDuLieu/
  README.md
  kiem_tra_du_lieu/
    KET_QUA.md          báo cáo bước 0
    scripts/            kiem_tra_video_va_trung.py, ma_tran_trung_8x8.py
    bang/               01–09_*.csv (từng ảnh, dấu hiệu video, ma trận 8×8, các cặp đã soi)
    hinh/               hình soi, soi_8x8/
    trung_lap/          bản chép các cặp trùng thật, danh_sach_trung.csv
  scripts/gom_anh_640.py   gom và thu nhỏ ảnh
  anh_gom/                 polyp/, am_tinh/, danh_sach_anh.csv (10.521 ảnh, cạnh dài ≤ 640)
```
