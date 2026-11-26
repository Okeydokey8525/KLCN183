# Hướng 2: kết quả trên tập test (10 seed ghép cặp)

Chạy ngày 18/11/2026 trên laptop (RTX 5050) bằng [`danh_gia/danh_gia_test_h2.py`](../../danh_gia/danh_gia_test_h2.py), suy luận 20 checkpoint mất 1,2 phút. Hình vẽ bằng [`danh_gia/ve_hinh_h2.py`](../../danh_gia/ve_hinh_h2.py). Log ở [`bang/_log_danh_gia.txt`](bang/_log_danh_gia.txt).

**Thiết kế.**
- Checkpoint `best.pt` được chọn trên **val mới** (120 ảnh polyp và 40 ảnh nền, tách từ train cũ).
- Tập **test** là đúng 160 ảnh val cũ trong báo cáo, và mỗi checkpoint chỉ được đánh giá trên tập này một lần.
- Giao thức giống Bảng 4.8: imgsz 640, conf 0,25, iou 0,7, `retina_masks`, FP32. Nhãn chính là polygon YOLO vẽ ở độ phân giải gốc.
- Endpoint chính, chọn từ trước, là **Dice theo pixel**. Các chỉ số còn lại là phụ và mang tính thăm dò, vì không hiệu chỉnh so sánh bội.

## 1. TSVM so với YOLO26s-seg (seed 0–9)

| Chỉ số | YOLO26s-seg | + TSVM | Δ (TSVM − YOLO) [CI 95%] | p t | p Wilcoxon | d_z | TSVM cao hơn |
|---|---|---|---|---:|---:|---:|---:|
| **Dice (endpoint chính)** | 0,850 ± 0,012 | 0,846 ± 0,018 | −0,0036 [−0,013; 0,006] | 0,43 | 0,28 | −0,26 | 2/10 |
| IoU | 0,794 ± 0,011 | 0,791 ± 0,016 | −0,0035 [−0,012; 0,005] | 0,39 | 0,19 | −0,29 | 2/10 |
| Precision | 0,860 ± 0,013 | 0,859 ± 0,023 | −0,0008 | 0,91 | 0,56 | −0,04 | 3/10 |
| Recall | 0,872 ± 0,013 | 0,864 ± 0,017 | −0,0078 [−0,014; −0,001] | 0,025 | 0,037 | −0,85 | 3/10 |
| Ảnh nền bị báo nhầm | 1,5% | 2,5% | +1,0 điểm % | 0,31 | 0,34 | | |
| mask mAP50–95 (test) | 0,707 ± 0,011 | 0,702 ± 0,005 | −0,0046 [−0,011; 0,002] | 0,16 | 0,049 | −0,48 | 1/10 |
| mask mAP50 (test) | 0,897 ± 0,013 | 0,895 ± 0,013 | −0,0019 | 0,68 | 0,63 | | 4/10 |
| box mAP50–95 (test) | 0,715 ± 0,012 | 0,708 ± 0,007 | −0,0076 | 0,057 | 0,084 | | 2/10 |

Mean ± SD qua 10 seed (SD tính với ddof = 1). Nguồn: [`bang/12_thong_ke.csv`](bang/12_thong_ke.csv), từng seed ở [`bang/11_tung_seed.csv`](bang/11_tung_seed.csv).

**Diễn giải.**
- Trên endpoint chính, **chưa đủ bằng chứng cho thấy TSVM khác YOLO26s-seg**. Dice của TSVM thấp hơn 0,004 theo xu hướng (p = 0,43); TSVM chỉ cao hơn ở 2/10 seed (seed 2 và 4).
- Recall của TSVM thấp hơn ở mức thăm dò: cả hai kiểm định có p < 0,05, nhưng đây là chỉ số phụ chưa hiệu chỉnh so sánh bội. Số ảnh polyp bị bỏ sót hoàn toàn là 6,8 ở TSVM so với 6,5 ở YOLO (trung bình mỗi seed).
- mask mAP50–95 có Wilcoxon p = 0,049 nhưng paired t p = 0,16. Hai kiểm định không thống nhất, nên đây chỉ là xu hướng TSVM thấp hơn.
- **Độ ổn định giữa các seed:** SD của mAP ở TSVM nhỏ hơn (0,005 so với 0,011), nhưng SD của Dice lại lớn hơn (0,018 so với 0,012). Kiểm định Levene không có ý nghĩa ở chỉ số nào (p ≥ 0,05), nên không kết luận được mô hình nào ổn định hơn.

## 2. So với báo cáo cũ (cùng 160 ảnh, seed 0–9)

Trong báo cáo cũ, checkpoint được chọn trên chính tập báo cáo. Ở hướng 2, checkpoint được chọn trên một tập val riêng. Bảng dưới so hai cách trên cùng 160 ảnh.

| Mô hình | Dice báo cáo cũ | Dice hướng 2 | Δ | p t |
|---|---|---|---:|---:|
| YOLO26s-seg | 0,850 ± 0,022 | 0,850 ± 0,012 | −0,0006 | 0,95 |
| + TSVM | 0,852 ± 0,014 | 0,846 ± 0,018 | −0,0060 | 0,41 |

Nguồn số cũ: `Test_Mau/00_tai_lieu_tong_quan/bang/pixel_640/10_pixel_tung_anh.csv`, chỉ lấy seed 0–9 (chỉ đọc), tính lại trong [`bang/13_bao_cao_cu_seed0_9.csv`](bang/13_bao_cao_cu_seed0_9.csv).

- Tách val riêng **không làm số thay đổi đáng kể**: Dice của YOLO gần như giữ nguyên, của TSVM giảm 0,006 nhưng không có ý nghĩa thống kê (p = 0,41).
- Theo số này, thiên lệch do chọn checkpoint trên chính tập báo cáo là nhỏ.
- Ở báo cáo cũ (seed 0–9), TSVM nhỉnh hơn YOLO 0,002. Ở hướng 2, TSVM thấp hơn 0,004. Cả hai mức chênh đều nằm trong dao động giữa các seed.
- Hai cách không hoàn toàn giống nhau: hướng 2 chỉ train trên 760 ảnh polyp thay vì 880. Như vậy khác biệt giữa hai cột gồm cả ảnh hưởng của việc bớt dữ liệu train, không chỉ của cách chọn checkpoint.

## 3. Chi phí huấn luyện (Kaggle T4)

| | YOLO26s-seg | + TSVM |
|---|---|---|
| Giờ mỗi seed | 2,11 ± 0,09 | 4,10 ± 0,06 |
| Epoch tốt nhất trên val (trung bình, min–max) | 88,6 (70–98) | 89,2 (79–99) |

## 4. Hình (`hinh/`, PNG 300 dpi và PDF)

| Hình | Nội dung |
|---|---|
| `h01_dice_tung_seed_test` | Dice từng seed, nối cặp theo seed, kèm mean ± SD |
| `h02_map_tung_seed_test` | mask mAP50–95 trên test, nối cặp theo seed |
| `h03_so_voi_bao_cao_cu` | Dice, IoU, tỉ lệ ảnh nền bị báo nhầm: báo cáo cũ so với hướng 2 |
| `h04_duong_cong_val` | mask mAP50–95 trên val mới theo epoch (trung bình, vùng min–max qua 10 seed) |
| `h05_minh_hoa_seed0` | 4 ảnh test ở seed 0: TSVM kém nhất, chênh ở mức trung vị, TSVM tốt nhất, YOLO kém nhất |

Hình chỉ số 10 seed (thêm ngày 24/11) được vẽ bằng [`../../../ve_hinh_chi_so_10seed.py`](../../../ve_hinh_chi_so_10seed.py) từ `results.csv` và `test_metrics.json`. Script không dùng GPU.

| Hình | Nội dung |
|---|---|
| `k01_val_mask_theo_epoch` | Precision, Recall, mAP50, mAP50–95 của mask trên val theo epoch (trung bình và min–max qua 10 seed) |
| `k02_val_box_theo_epoch` | Giống k01, cho box |
| `k03_loss_theo_epoch` | Loss train và val (box, seg, cls). Trục cắt từ epoch 3 vì 2 epoch đầu có loss rất lớn. Bước hạ ở epoch 91 là lúc tắt mosaic |
| `k04_test_mask_tung_seed` | 4 chỉ số mask trên test, nối cặp theo seed, ghi Δ, p, số seed TSVM cao hơn |
| `k05_cot_test_ultralytics` | Cột 8 chỉ số Ultralytics (mask và box) trên test, mean ± SD, có chấm từng seed |
| `k06_cot_pixel` | Cột Dice, IoU, Precision, Recall theo pixel |
| `k07_chi_phi_tung_seed` | Giờ huấn luyện và epoch tốt nhất của từng seed |

Ghi chú về `k07`:
- Giờ huấn luyện tính từ cột `time` của `results.csv`: YOLO 2,10, TSVM 4,08. Bảng ở mục 3 lấy số từ `test_metrics.json` (2,11 và 4,10), có tính thêm thời gian ngoài vòng huấn luyện.
- Epoch tốt nhất trong `k07` tính theo fitness Ultralytics (box + mask): YOLO 91,3, TSVM 91,2. Mục 3 ghi 88,6 và 89,2 vì chỉ tính theo mask, nên hai bên lệch nhau.

### Hình quá trình huấn luyện từng seed (train/val, không dùng test)

Các hình nằm ở [`../hinh_huan_luyen/`](../hinh_huan_luyen/), vẽ bằng [`../../../ve_hinh_huan_luyen_tung_seed.py`](../../../ve_hinh_huan_luyen_tung_seed.py) từ `results.csv`. Bảng số: `hinh_huan_luyen/val_tai_epoch_tot_nhat.csv`.

| Hình | Nội dung |
|---|---|
| **`r00_ket_qua_huan_luyen_tb_10seed`** | **Hình tổng hợp kiểu `results.png`.** Mỗi ô là trung bình 10 seed của YOLO và TSVM, vẽ chồng lên nhau, có vùng min–max. Các ô gồm: train loss (box, seg, cls, L1, sem), val loss (box, seg, cls, L1), Precision, Recall, mAP50, mAP50–95 trên val cho mask và box. Bỏ val sem loss vì Ultralytics luôn ghi bằng 0 |
| `r_tung_bieu_do/r01`–`r17` | 17 biểu đồ của `r00`, mỗi biểu đồ lưu thành một hình riêng để đưa lên slide: r01–r05 train loss, r06–r09 val loss, r10–r13 chỉ số mask trên val, r14–r17 chỉ số box trên val |
| `t01`–`t04` | Precision, Recall, mAP50, mAP50–95 (mask) trên val theo epoch. Mỗi ô là 1 seed, có YOLO và TSVM, chấm đánh dấu epoch tốt nhất (epoch lưu `best.pt`) |
| `t05` | Seg loss train (nét liền) và val (nét đứt) theo epoch, từng seed |
| `t06` | Cột chỉ số mask trên val tại epoch tốt nhất, từng seed, kèm cột TB ± SD |
| `t07` | Giống `t06`, cho box |
| `t08` | Cột loss ở epoch cuối (100), từng seed. Với loss, "TSVM cao hơn" nghĩa là TSVM tệ hơn |

Trên val (160 ảnh) tại epoch tốt nhất, YOLO so với TSVM:

| Chỉ số | YOLO | TSVM | Δ | p t | Số seed TSVM cao hơn |
|---|---|---|---|---|---|
| mask mAP50–95 | 0,798 | 0,796 | −0,0017 | 0,69 | 6/10 |
| mask mAP50 | 0,959 | 0,955 | −0,0046 | 0,08 | 3/10 |
| Precision | 0,955 | 0,946 | −0,0089 | 0,47 | 5/10 |
| Recall | 0,904 | 0,904 | +0,0001 | 1,00 | 6/10 |

Chưa có chỉ số nào khác biệt có ý nghĩa.

Từ epoch 60 trở đi, val seg loss đi ngang trong khi train loss vẫn giảm. Đây là dấu hiệu bắt đầu quá khớp nhẹ, và cả hai mô hình đều như vậy.

Trục tung của các hình cột không bắt đầu từ 0, để thấy rõ chênh lệch nhỏ. Khi trình bày cần đọc số, không so chiều cao cột.

## 5. Câu nói gợi ý khi bảo vệ

> "Để tránh thiên lệch do chọn checkpoint trên chính tập báo cáo, nhóm chia lại BG20 thành train/val/test và huấn luyện lại 10 seed cho cả hai mô hình. Trên tập test riêng, Dice của TSVM là 0,846 ± 0,018 so với 0,850 ± 0,012 của mô hình gốc (p = 0,43). Chưa đủ bằng chứng để nói TSVM khác mô hình gốc, và kết quả này nhất quán với báo cáo."

## 6. Giới hạn

- Có 2 cặp ảnh train–test gần trùng (tương quan ≥ 0,95). Các cặp này có sẵn từ split của tác giả Kvasir-SEG, xem `../../du_lieu/kiem_tra_gan_trung.csv`.
- Tập test nhỏ (120 ảnh polyp, 40 ảnh nền), cùng nguồn Kvasir.
- Với 10 seed, Wilcoxon hai phía chỉ có p nhỏ nhất là 0,002. Các chỉ số phụ chưa hiệu chỉnh so sánh bội.

Các file có hậu tố `_thu` là của lần chạy thử với seed 0, giữ lại để đối chiếu.
