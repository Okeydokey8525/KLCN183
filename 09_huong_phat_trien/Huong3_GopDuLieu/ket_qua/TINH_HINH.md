# Kết quả hướng 3 (gom 22/11, đủ 10 seed ngày 24/11/2026)

Tạo bằng [`../scripts/gom_ket_qua_h3.py`](../scripts/gom_ket_qua_h3.py).

Số đo test lấy từ `test_metrics.json`, do notebook Kaggle tính trên `best.pt` (checkpoint chọn theo val). Tập test có 522 ảnh: 448 ảnh polyp và 74 ảnh âm tính.

- `zip_tai_ve/`: 20 zip gốc tải từ Kaggle.
- `H3_Gop_3tap_<model>_s<seed>/`: thư mục giải nén của từng lượt.
- `bang_test_tung_seed.csv`: bảng số đo từng seed. Các bản trước nằm ở `.bak_*`.

## Tình trạng: đủ 20/20 lượt, tất cả đủ 100 epoch

| Mô hình | Seed | Giờ huấn luyện/seed (T4, từ results.csv) |
|---|---|---|
| YOLO26s-seg | 0–9 | 5,56–5,91 (TB 5,72) |
| TSVM_FP32 | 0–9 | 11,14–11,69 (TB 11,40) |

Về 4 seed TSVM s1, s3, s6, s8:
- Lần chạy đầu dừng an toàn ở epoch 98–99. Sau đó đã chạy tiếp đủ 100 epoch bằng `h3-chay-tiep-4seed` (ngày 23/11), nên `test_metrics.json` có `chay_tiep = true`.
- 98–99 epoch đầu trong `results.csv` giống hệt bản cũ.
- Ở cả 4 seed, 1–2 epoch chạy thêm không vượt epoch tốt nhất trên val, nên `best.pt` và số đo test không đổi.
- Bản dừng an toàn (thư mục và zip) đã chuyển sang `_Luu_Tru_Ngoai\Baseline_PVT_VMUNet\Huong3_GopDuLieu\ket_qua\…_dung_an_toan`, có ghi nhật ký.

## Mask mAP50-95 trên test (10 seed ghép cặp, số Ultralytics)

| Seed | YOLO26s-seg | TSVM_FP32 |
|---|---|---|
| 0 | 0,7136 | 0,7094 |
| 1 | 0,7107 | 0,7132 |
| 2 | 0,7160 | 0,7154 |
| 3 | 0,6909 | 0,7164 |
| 4 | 0,7026 | 0,7012 |
| 5 | 0,7079 | 0,7111 |
| 6 | 0,7113 | 0,7169 |
| 7 | 0,7091 | 0,7105 |
| 8 | 0,7058 | 0,7023 |
| 9 | 0,7032 | 0,7089 |
| TB ± SD | 0,7071 ± 0,0071 | 0,7105 ± 0,0054 |

| Chỉ số (test) | Δ (TSVM − YOLO) [CI 95%] | p t | p Wilcoxon | d_z | TSVM cao hơn |
|---|---|---:|---:|---:|---:|
| mask mAP50-95 | +0,0034 [−0,0026; +0,0095] | 0,234 | 0,275 | 0,40 | 6/10 |
| mask mAP50 | +0,0015 [−0,0023; +0,0052] | 0,397 | 0,496 | 0,28 | 5/10 |
| box mAP50-95 | +0,0009 [−0,0086; +0,0103] | 0,836 | 0,432 | 0,07 | 3/10 |

Kết luận:
- Trung bình của TSVM nhỉnh hơn một chút, nhưng chưa đủ bằng chứng về khác biệt.
- SD của TSVM nhỏ hơn (0,0054 so với 0,0071), nhưng kiểm định Brown–Forsythe cho p = 0,62 nên cũng chưa đủ bằng chứng.

Đánh giá Dice/IoU theo pixel trên test và trên 3 bộ ngoài (CVC-ClinicDB, CVC-ColonDB, ETIS) đã xong ngày 24/11. Kết quả ở [`danh_gia/KET_QUA.md`](danh_gia/KET_QUA.md).
