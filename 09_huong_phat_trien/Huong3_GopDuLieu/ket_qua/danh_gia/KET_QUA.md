# Hướng 3: kết quả theo pixel trên test gộp và 3 bộ ngoài (10 seed ghép cặp)

Chạy ngày 24/11/2026 trên laptop (RTX 5050) bằng [`../../danh_gia/danh_gia_h3.py`](../../danh_gia/danh_gia_h3.py). Suy luận 20 checkpoint trên 4 tập mất 11,3 phút. Log: [`log_danh_gia_h3.txt`](log_danh_gia_h3.txt).

**Thiết kế.**
- Dữ liệu gộp (Kvasir, PolypGen, PolypDB Simula/Karolinska, ảnh âm tính) chia 3 tập: train 2.455, val 532, test 522 (448 ảnh polyp, 74 ảnh âm tính).
- Checkpoint `best.pt` được chọn trên val. Test và 3 bộ ngoài chỉ được đánh giá một lần.
- Bộ ngoài: CVC-ClinicDB 612 ảnh, CVC-ColonDB 380 ảnh, ETIS-LaribPolypDB 196 ảnh. Cả 3 bộ đều không dùng khi huấn luyện, và đã kiểm tra là không trùng với bộ gộp. PolypGen không dùng làm bộ ngoài vì nó nằm trong dữ liệu gộp.
- Giao thức giống Bảng 4.8 và hướng 2: imgsz 640, conf 0,25, iou 0,7, `retina_masks`, FP32, hợp các mặt nạ dự đoán của mỗi ảnh.
- Nhãn test chính là polygon YOLO vẽ ở độ phân giải gốc. Bộ ngoài dùng mặt nạ gốc của tác giả.
- Endpoint chính, chọn từ trước, là **Dice theo pixel trên test gộp**. Dice được tính cho từng ảnh polyp rồi lấy trung bình (ảnh không có dự đoán thì Dice = 0). Các chỉ số khác là phụ và mang tính thăm dò, vì không hiệu chỉnh so sánh bội.

## 1. Test gộp (522 ảnh): TSVM so với YOLO26s-seg

| Chỉ số | YOLO26s-seg | + TSVM | Δ (TSVM − YOLO) [CI 95%] | p t | p Wilcoxon | d_z | TSVM cao hơn |
|---|---|---|---|---:|---:|---:|---:|
| **Dice (endpoint chính)** | 0,839 ± 0,007 | 0,839 ± 0,009 | −0,0006 [−0,007; 0,006] | 0,84 | 1,00 | −0,07 | 5/10 |
| IoU | 0,788 ± 0,007 | 0,789 ± 0,008 | +0,0008 [−0,005; 0,006] | 0,77 | 0,38 | 0,10 | 7/10 |
| Precision | 0,852 ± 0,008 | 0,851 ± 0,010 | −0,0013 | 0,70 | 0,70 | −0,12 | 4/10 |
| Recall | 0,853 ± 0,007 | 0,851 ± 0,008 | −0,0012 | 0,59 | 0,85 | −0,18 | 5/10 |
| Dice gộp pixel (micro) | 0,891 ± 0,007 | 0,893 ± 0,006 | +0,0020 | 0,49 | 0,56 | 0,23 | 5/10 |
| Dice theo mặt nạ gốc | 0,838 ± 0,007 | 0,837 ± 0,009 | −0,0005 | 0,85 | 1,00 | | 5/10 |
| Ảnh polyp bị bỏ sót hoàn toàn | 6,9% | 7,4% | +0,5 điểm % | 0,22 | 0,26 | | |
| Ảnh âm tính bị báo nhầm | 11,6% | 11,5% | −0,1 điểm % | 0,87 | 0,92 | | |
| mask mAP50–95 | 0,707 ± 0,007 | 0,711 ± 0,005 | +0,0034 [−0,003; 0,010] | 0,23 | 0,28 | 0,41 | 6/10 |
| mask mAP50 | 0,893 ± 0,004 | 0,895 ± 0,006 | +0,0015 | 0,40 | 0,49 | | 6/10 |
| box mAP50–95 | 0,704 ± 0,010 | 0,705 ± 0,007 | +0,0009 | 0,83 | 0,43 | | 3/10 |

Mean ± SD qua 10 seed (ddof = 1). Nguồn: [`bang/12_thong_ke.csv`](bang/12_thong_ke.csv), từng seed ở [`bang/11_tung_seed.csv`](bang/11_tung_seed.csv).

Dice từng seed trên test:

| Seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO | 0,842 | 0,841 | 0,847 | 0,838 | 0,829 | 0,852 | 0,839 | 0,837 | 0,836 | 0,829 |
| TSVM | 0,844 | 0,840 | 0,856 | 0,833 | 0,833 | 0,838 | 0,841 | 0,836 | 0,823 | 0,842 |

**Diễn giải.**
- Trên endpoint chính, hai mô hình gần như bằng nhau: Δ = −0,0006, p = 0,84, TSVM cao hơn ở 5/10 seed. **Chưa đủ bằng chứng để nói TSVM khác YOLO26s-seg.**
- mask mAP50–95 có xu hướng nhỉnh về TSVM (+0,003, 6/10 seed) nhưng không có ý nghĩa thống kê.
- Nhãn polygon và mặt nạ gốc cho kết luận giống nhau (Δ −0,0006 và −0,0005).
- **Độ ổn định:** SD của Dice giữa các seed là 0,009 (TSVM) so với 0,007 (YOLO). Brown–Forsythe p = 0,69, nên không kết luận được mô hình nào ổn định hơn.

## 2. Ba bộ ngoài (không có trong dữ liệu huấn luyện)

| Bộ | Chỉ số | YOLO26s-seg | + TSVM | Δ [CI 95%] | p t | p Wilcoxon | TSVM cao hơn |
|---|---|---|---|---|---:|---:|---:|
| CVC-ClinicDB (612) | Dice | 0,850 ± 0,004 | 0,845 ± 0,006 | −0,0043 [−0,010; 0,001] | 0,11 | 0,16 | 4/10 |
| | Bỏ sót ảnh | 5,4% | 5,8% | +0,4 điểm % | 0,20 | 0,22 | |
| CVC-ColonDB (380) | Dice | 0,694 ± 0,008 | 0,687 ± 0,011 | −0,0070 [−0,018; 0,004] | 0,19 | 0,28 | 3/10 |
| | Bỏ sót ảnh | 19,0% | 20,2% | +1,3 điểm % [0,1; 2,5] | 0,035 | 0,063 | |
| ETIS (196) | Dice | 0,752 ± 0,023 | 0,750 ± 0,023 | −0,0022 [−0,024; 0,020] | 0,83 | 0,63 | 3/10 |
| | Bỏ sót ảnh | 13,8% | 13,8% | 0 | 1,00 | 0,85 | |

**Diễn giải.**
- Ở cả 3 bộ ngoài, Dice của TSVM thấp hơn YOLO một chút (0,002–0,007), nhưng không bộ nào có ý nghĩa thống kê. Đây chỉ là xu hướng.
- CVC-ColonDB khó nhất (Dice khoảng 0,69, bỏ sót khoảng 1/5 ảnh). Tỉ lệ bỏ sót của TSVM cao hơn ở mức thăm dò (p t 0,035, nhưng Wilcoxon 0,063, chỉ số phụ chưa hiệu chỉnh so sánh bội).
- ETIS có SD giữa các seed lớn nhất (khoảng 0,023), vì bộ chỉ có 196 ảnh và nhiều polyp nhỏ.

## 3. Theo nguồn ảnh trong test (thăm dò)

| Nguồn | Số ảnh | YOLO Dice | TSVM Dice | Δ | p t | TSVM cao hơn |
|---|---:|---|---|---:|---:|---:|
| Kvasir (KV) | 150 | 0,878 | 0,881 | +0,003 | 0,67 | 5/10 |
| PolypGen (PG) | 209 | 0,797 | 0,792 | −0,004 | 0,43 | 5/10 |
| PolypDB Simula WLI (SW) | 64 | 0,917 | 0,911 | −0,006 | 0,23 | 3/10 |
| PolypDB Simula NBI (SN) | 21 | 0,727 | 0,756 | +0,029 | 0,14 | 7/10 |
| PolypDB Karolinska (KA) | 4 | 0,957 | 0,945 | −0,012 | 0,38 | 6/10 |

Ảnh âm tính bị báo nhầm, chia theo nguồn:

| Nguồn | YOLO | TSVM |
|---|---|---|
| Cecum (CE, 30 ảnh) | 7,7% | 7,7% |
| PolypGen mặt nạ rỗng (PGR, 20 ảnh) | 30,5% | 28,5% |
| PolypGen chuỗi âm tính (PGS, 24 ảnh) | 0,8% | 2,1% |

Nguồn: [`bang/14_theo_nguon_test.csv`](bang/14_theo_nguon_test.csv) và [`bang/15_thong_ke_theo_nguon_test.csv`](bang/15_thong_ke_theo_nguon_test.csv).

- Nhóm SN (21 ảnh) và KA (4 ảnh) quá nhỏ để kết luận.
- Nhóm PolypGen khó nhất trong test (Dice khoảng 0,79).
- Ảnh PGR bị báo nhầm nhiều. Có thể do ảnh lấy từ chuỗi video có polyp nhưng không được vẽ mặt nạ; điều này chưa kiểm tra bằng mắt.

## 4. Chi phí huấn luyện (Kaggle T4)

| | YOLO26s-seg | + TSVM |
|---|---|---|
| Giờ mỗi seed | 5,72 (5,56–5,91) | 11,40 (11,14–11,69) |

Nguồn: [`../bang_test_tung_seed.csv`](../bang_test_tung_seed.csv).

## 5. Hình (`hinh/`, PNG 300 dpi và PDF)

Vẽ bằng [`../../danh_gia/ve_hinh_h3.py`](../../danh_gia/ve_hinh_h3.py) từ các bảng ở trên. Script không suy luận lại và không dùng GPU.

| Hình | Nội dung |
|---|---|
| `h01_dice_tung_seed_4_tap` | Dice từng seed, nối cặp theo seed, kèm mean ± SD, cho 4 tập (test và 3 bộ ngoài) |
| `h02_map_tung_seed_test` | mask mAP50–95 trên test, nối cặp theo seed |
| `h03_delta_dice_theo_tap` | Δ Dice (TSVM − YOLO) kèm CI 95% cho 4 tập và 5 nguồn ảnh trong test. Đây là hình gọn nhất để đưa lên slide |
| `h04_duong_cong_val` | mask mAP50–95 trên val theo epoch (trung bình và vùng min–max qua 10 seed). Hai đường gần như trùng nhau |
| `h05_minh_hoa_test_seed0` | 4 ảnh test ở seed 0: TSVM kém hơn nhiều nhất, chênh ở mức trung vị (cả hai bỏ sót), TSVM tốt hơn nhiều nhất, YOLO kém nhất |
| `h06_minh_hoa_bo_ngoai_seed0` | Mỗi bộ ngoài 2 ảnh: ảnh TSVM kém hơn nhiều nhất và ảnh TSVM tốt hơn nhiều nhất. Các ca này được chọn có chủ đích nên chỉ để minh họa; hai chiều đều có ca bỏ sót hoàn toàn |

Hình chỉ số 10 seed được vẽ bằng [`../../../ve_hinh_chi_so_10seed.py`](../../../ve_hinh_chi_so_10seed.py) từ `results.csv` và `test_metrics.json`. Script không dùng GPU.

| Hình | Nội dung |
|---|---|
| `k01_val_mask_theo_epoch` | Precision, Recall, mAP50, mAP50–95 của mask trên val theo epoch (trung bình và min–max qua 10 seed). Hai đường gần như trùng nhau |
| `k02_val_box_theo_epoch` | Giống k01, cho box |
| `k03_loss_theo_epoch` | Loss train và val (box, seg, cls). Trục cắt từ epoch 3. Bước hạ ở epoch 91 là lúc tắt mosaic |
| `k04_test_mask_tung_seed` | 4 chỉ số mask trên test, nối cặp theo seed. Precision của TSVM cao hơn 0,013 (p 0,12; 7/10 seed), Recall thấp hơn 0,004 (p 0,47) |
| `k05_cot_test_ultralytics` | Cột 8 chỉ số Ultralytics (mask và box) trên test, mean ± SD, có chấm từng seed |
| `k06_cot_pixel` | Cột Dice, IoU, Precision, Recall theo pixel, cho test và 3 bộ ngoài |
| `k07_chi_phi_tung_seed` | Giờ huấn luyện (YOLO 5,72, TSVM 11,40) và epoch tốt nhất theo fitness Ultralytics (90,2 và 91,4) |

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

Trên val (532 ảnh) tại epoch tốt nhất, YOLO so với TSVM:

| Chỉ số | YOLO | TSVM | Δ | p t | Số seed TSVM cao hơn |
|---|---|---|---|---|---|
| mask mAP50–95 | 0,689 | 0,690 | +0,0014 | 0,57 | 6/10 |
| mask mAP50 | 0,884 | 0,882 | −0,0019 | 0,35 | 3/10 |
| Precision | 0,898 | 0,894 | −0,0043 | 0,54 | 4/10 |
| Recall | 0,791 | 0,801 | +0,0105 | 0,09 | 8/10 |

Chưa có chỉ số nào khác biệt có ý nghĩa. Recall của TSVM có xu hướng cao hơn (8/10 seed).

Train box loss ở epoch cuối của TSVM cao hơn 0,005 (p 0,02, thăm dò). Đây là loss trên tập train, không phải chỉ số chất lượng.

Trục tung của các hình cột không bắt đầu từ 0, để thấy rõ chênh lệch nhỏ. Khi trình bày cần đọc số, không so chiều cao cột.

## 6. Câu nói gợi ý khi bảo vệ

> "Để kiểm tra trên dữ liệu đa dạng hơn, nhóm gộp Kvasir, PolypGen và PolypDB, thêm ảnh âm tính, chia train/val/test và huấn luyện lại 10 seed cho cả hai mô hình. Trên 522 ảnh test, Dice của hai mô hình đều khoảng 0,839 (p = 0,84). Trên 3 bộ ngoài CVC-ClinicDB, CVC-ColonDB và ETIS, TSVM thấp hơn 0,002–0,007 nhưng không có ý nghĩa thống kê. Nhóm kết luận chưa đủ bằng chứng để nói TSVM khác mô hình gốc về độ chính xác, trong khi thời gian huấn luyện gấp khoảng 2 lần."

## 7. Giới hạn

- Có 10 seed. Wilcoxon hai phía với n = 10 có p nhỏ nhất là 0,002. Chỉ số phụ và phân tích theo nguồn chưa hiệu chỉnh so sánh bội.
- Bộ ngoài là ảnh trích từ video, nên nhiều ảnh trong cùng một bộ gần giống nhau. Số ảnh hiệu dụng vì vậy nhỏ hơn số ảnh thật.
- Nhãn test chính là polygon YOLO. So với mặt nạ gốc, Dice trung bình là 0,989 (thấp nhất 0,954), và kết luận không đổi khi dùng mặt nạ gốc.
- Hình minh họa chỉ dùng seed 0 (mặt nạ dự đoán lưu ở `du_lieu_dinh_tinh/`).
