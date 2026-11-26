# Hướng 2: BG20 chia lại thành 3 tập, 10 seed YOLO26s-seg và TSVM

Chốt ngày 17/11/2026. Bảo vệ ngày 27/11/2026. Kết quả chỉ dùng cho slide và phần trả lời hội đồng, không sửa báo cáo đã nộp. Hướng 3 (gộp dữ liệu rồi chia 3 tập, cũng 10 seed YOLO + TSVM) làm **sau** khi xong hướng này.

## 1. Vì sao

Trong báo cáo, checkpoint được chọn trên tập val và số liệu cũng báo cáo trên chính tập val đó (nhận xét II.2 của GVHD). Hướng 2 tách một tập val riêng để chọn checkpoint, còn tập test chỉ được dùng một lần để báo cáo.

## 2. Cách chia (đã thực hiện)

Script [`scripts/tao_chia_3_tap.py`](scripts/tao_chia_3_tap.py), seed chia cố định `2026`, chạy một lần trước khi huấn luyện.

| Tập | Ảnh polyp | Ảnh nền | Số polyp | Nguồn |
|---|---:|---:|---:|---|
| Train | 760 | 120 | 812 | train cũ |
| Val (chọn checkpoint) | 120 | 40 | 124 | tách từ train cũ: 40 nhỏ, 40 vừa, 40 lớn (tertile diện tích polyp) và 40 ảnh nền |
| **Test** (chỉ báo cáo) | 120 | 40 | 127 | **đúng tập val cũ** (split 880/120 của tác giả Kvasir-SEG) |

Kiểm tra (kết quả ở [`du_lieu/`](du_lieu/)):
- md5 không lặp giữa các tập.
- Ảnh gần trùng (tương quan ảnh xám 64×64 sau cắt viền đen, cùng cách với kiểm tra PolypGen): trong train cũ có 5 cặp từ 0,95 trở lên. Mỗi cặp được gom thành cụm và cho vào cùng một tập, nên **train–val: 0 cặp** từ 0,95 trở lên.
- **train–test: 2 cặp** từ 0,95 trở lên. Hai cặp này có sẵn từ split gốc của tác giả, không đổi vì test phải giữ nguyên. Ghi vào phần giới hạn.
- val–test: 0 cặp.

Tập test trùng tập val cũ, nên số mới so được với Bảng 4.8 trên cùng 160 ảnh. Khác biệt duy nhất là lúc chọn checkpoint, mô hình không còn nhìn thấy tập test (và train còn 760 thay vì 880 ảnh polyp).

## 3. Huấn luyện

| | |
|---|---|
| Mô hình | YOLO26s-seg (baseline), YOLO26s-seg + TSVM (tầng 10 C2TSVMamba, 1.811.201 tham số). Chỉ hai mô hình này |
| Seed | 0–9, chốt trước |
| Siêu tham số | Như 40 lượt trong báo cáo: 640, 100 epoch, batch 8, AdamW lr0 1e-3, warmup 5, close_mosaic 10, amp=False, deterministic, patience 100, workers 2 |
| Mã | Ultralytics 8.4.127 bản `ultralytics_TSVM_FP32` cho cả hai mô hình. Baseline trong báo cáo dùng pip 8.4.127: mã huấn luyện giống nhau, chỉ khác `best.pt` lưu FP32 |
| Chọn checkpoint | `best.pt` theo fitness trên **val** (như mặc định Ultralytics) |
| Ngân sách | Khoảng 65–70 giờ GPU T4, 8 phiên. Hướng dẫn: [`Goi_Kaggle_H2/HUONG_DAN.md`](Goi_Kaggle_H2/HUONG_DAN.md) |

## 4. Đánh giá (sau khi có 20 checkpoint)

- **Chỉ trên tập test**, mỗi checkpoint một lần:
  - mAP box và mask: notebook tự tính, lưu ở `test_metrics.json`;
  - chỉ số theo pixel như Bảng 4.8: chạy trên laptop bằng script mới trong `danh_gia/`, chép giao thức từ `danh_gia_pixel.py`.
- So ghép cặp TSVM với YOLO trên 10 seed:
  - mean ± SD, Δ và CI 95%, paired t, Wilcoxon (với 10 seed, p thấp nhất được 0,002), Cohen's d_z, số seed thắng;
  - endpoint chính chọn trước là **Dice theo pixel trên test**, các chỉ số khác là phụ.
- Đặt cạnh kết quả cũ (chọn checkpoint trên chính tập báo cáo, seed 0–9 của báo cáo) để xem việc tách val làm số thay đổi bao nhiêu.
- Viết "xu hướng" khi p ≥ 0,05, không viết "vượt trội".

## 5. Lịch

| Ngày | Việc | Ai làm |
|---|---|---|
| 17/11 | Chia dữ liệu, kiểm tra trùng, đóng gói, sinh notebook | Claude (xong) |
| khi sẵn sàng | Tải dataset lên, chia sẻ, chạy 2 notebook chạy thử | Người dùng, gửi log cho Claude kiểm tra |
| khi có giờ GPU | 8 phiên trên 6 tài khoản (quota reset thứ Bảy 21/11) | Người dùng |
| 18/11 | Đánh giá trên test, thống kê, vẽ hình: **xong**, xem [`ket_qua/danh_gia/KET_QUA.md`](ket_qua/danh_gia/KET_QUA.md) | Claude |
| trước 27/11 | Câu trả lời hội đồng, hình cho slide (chưa đưa lên Canva) | Cùng làm |

## 6. Thư mục

```
Huong2_Chia3Tap/
  KE_HOACH.md                      file này
  scripts/tao_chia_3_tap.py        chia dữ liệu và kiểm tra trùng
  scripts/tao_notebook_h2.py       sinh notebook Kaggle
  du_lieu/Kvasir_BG20_3tap/        ảnh, nhãn, data.yaml
  du_lieu/danh_sach_chia.csv       từng ảnh: tập cũ, tập mới, nhóm kích thước, cụm, md5
  du_lieu/kiem_tra_gan_trung.csv   cặp ảnh tương quan >= 0,90 giữa các tập
  Goi_Kaggle_H2/                   zip dataset, notebook, HUONG_DAN.md
  ket_qua/YOLO26s_seg/             10 lượt H2_BG20_3tap_YOLO26s_seg_s0…s9 (giải nén 18/11)
  ket_qua/TSVM_FP32/               10 lượt H2_BG20_3tap_TSVM_FP32_s0…s9
  ket_qua/cau_hinh_kaggle/         data_bg20_3tap.yaml, yaml TSVM dùng trên Kaggle
```
