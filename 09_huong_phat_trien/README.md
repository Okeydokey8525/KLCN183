# 09. Hướng phát triển sau khi nộp khóa luận

Nhánh `huong-phat-trien` chỉ thêm thư mục này vào nội dung của `main`. Các thư mục `00`–`08` giữ nguyên như bản đã nộp ngày 11/11/2026.

Các thí nghiệm ở đây làm từ 14/11 đến 26/11/2026. Mục đích là chuẩn bị slide, trả lời hội đồng và làm hướng phát triển. Kết quả **không** được đưa vào báo cáo đã nộp.

Thư mục này chỉ chứa **kết quả**: mã/script, bảng CSV/JSON, hình, `results.csv` và `args.yaml` của từng lượt huấn luyện, cùng các tệp markdown. Những thứ **không** đưa lên vì quá nặng:
- ảnh dữ liệu;
- trọng số `.pt`/`.pth`;
- tệp zip tải từ Kaggle;
- wheel `mamba_ssm`;
- mã gốc Polyp-PVT và VM-UNet.

Danh sách tệp được lọc bằng [`chep_sang_nhanh_git.py`](chep_sang_nhanh_git.py). Bản đầy đủ nằm trên máy, ở thư mục `DeepLearning/Baseline_PVT_VMUNet/`.

Đường dẫn trong một số script trỏ tới bản đầy đủ trên máy, ví dụ `Huong*/du_lieu/` và `../Test_Mau`. Trên nhánh này, các script chủ yếu để đọc cách tính, không chạy lại trực tiếp được.

## Tóm tắt 4 hướng

| Hướng | Nội dung | Kết quả chính | Đọc chi tiết |
|---|---|---|---|
| 1. Polyp-PVT, VM-UNet | Thêm một mô hình Transformer và một mô hình Mamba chuyên phân đoạn polyp; BG20, ảnh 640, 5 seed | Dice pixel trên test: Polyp-PVT 0,894, PraNet 0,885, YOLO 0,854, TSVM 0,854, VM-UNet 0,739, U-Net 0,697. Polyp-PVT tốn 13,9 giờ huấn luyện mỗi seed | [`Huong1_PVT_VMUNet/ket_qua/so_sanh_6_mo_hinh/KET_QUA.md`](Huong1_PVT_VMUNet/ket_qua/so_sanh_6_mo_hinh/KET_QUA.md) |
| 2. BG20 chia 3 tập | Tách thêm một tập test riêng; 10 seed YOLO26s-seg và TSVM | Dice test: YOLO 0,850 ± 0,012, TSVM 0,846 ± 0,018 (p = 0,43, chưa đủ bằng chứng khác nhau) | [`Huong2_Chia3Tap/ket_qua/danh_gia/KET_QUA.md`](Huong2_Chia3Tap/ket_qua/danh_gia/KET_QUA.md) |
| 3. Gộp dữ liệu rồi chia 3 tập | Gộp nhiều bộ polyp, chia 3 tập; 10 seed mỗi mô hình | Đánh giá pixel trên test và 3 bộ ngoài | [`Huong3_GopDuLieu/ket_qua/danh_gia/KET_QUA.md`](Huong3_GopDuLieu/ket_qua/danh_gia/KET_QUA.md) |
| 4. SS2D quét bằng kernel CUDA | Thay phép quét PyTorch bằng `selective_scan` của `mamba_ssm`; giữ nguyên mô hình; 20 seed | Giờ huấn luyện mỗi seed giảm từ 4,68 xuống 2,46 (−47%). mAP50-95: 0,7232 → 0,7212 (p = 0,25). Không khác biệt có ý nghĩa trên test, dữ liệu ngoài và ảnh manh tràng | [`Huong4_KernelCUDA/ket_qua_moc3/KET_QUA.md`](Huong4_KernelCUDA/ket_qua_moc3/KET_QUA.md) |

So sánh chung hướng 2 và hướng 3 với báo cáo: [`so_sanh_bao_cao_H2_H3/`](so_sanh_bao_cao_H2_H3/).

## Cây thư mục

```
09_huong_phat_trien/
  README.md, chep_sang_nhanh_git.py, ve_hinh_*.py
  Huong1_PVT_VMUNet/      README, notebooks, goi_kaggle (không zip), danh_gia/, ket_qua/ (bảng, hình)
  Huong2_Chia3Tap/        KE_HOACH, scripts/, Goi_Kaggle_H2/, ket_qua/ (results.csv từng seed, danh_gia/, hình)
  Huong3_GopDuLieu/       README, scripts/, kiem_tra_du_lieu/, ket_qua/ (results.csv từng seed, danh_gia/, hình)
  Huong4_KernelCUDA/
    README.md             mục tiêu, thay đổi trong mã, mốc 1–3
    ma_nguon/             bản Ultralytics TSVM có nhánh quét kernel CUDA (chỉ khác topology_shape_vmamba.py)
    kiem_tra/             test tương đương trên laptop (mốc 1)
    Goi_Kaggle_H4/        notebook Kaggle mốc 2, mốc 3 và hướng dẫn
    ket_qua_moc2_kaggle/  đo kernel trên Tesla T4 (độ trễ, bộ nhớ, tốc độ huấn luyện)
    ket_qua_moc3/         20 seed: KET_QUA.md, bảng, pixel_test/, du_lieu_ngoai/, hinh/ (mỗi biểu đồ một tệp)
    scripts/              tạo notebook, phân tích, vẽ hình
```

## Hình hướng 4 (`Huong4_KernelCUDA/ket_qua_moc3/hinh/`)

| Tệp | Nội dung |
|---|---|
| `h4_01a_val_map_20seed` | mask mAP50-95 trên val, 20 seed: YOLO, TSVM PyTorch, TSVM CUDA |
| `h4_01b_gio_huan_luyen` | Giờ huấn luyện mỗi seed |
| `h4_01c_dice_test_20seed` | Dice pixel trên tập test Kvasir-SEG |
| `h4_01d_dice_du_lieu_ngoai` | Dice trên CVC-ClinicDB, CVC-ColonDB, ETIS, PolypGen |
| `h4_01e_bao_nham_anh_khong_polyp` | Tỉ lệ báo nhầm trên ảnh nền test, 800 ảnh manh tràng, ảnh PolypGen không polyp |
| `h4_01f_chenh_lech_cuda_pytorch` | Δ CUDA − PyTorch ghép cặp theo seed, kèm CI 95% |
| `h4_02a_dice_7_mo_hinh` | Dice test của 7 mô hình (6 mô hình hướng 1 và TSVM CUDA, seed 0–4) |
| `h4_02b_gio_7_mo_hinh` | Giờ huấn luyện của 7 mô hình |
| `h4_02c_dice_theo_gio_7_mo_hinh` | Dice theo giờ huấn luyện |

## Quy ước viết

- Dùng dấu phẩy thập phân.
- Khi p ≥ 0,05 thì chỉ viết "xu hướng" hoặc "chưa đủ bằng chứng", không viết "vượt trội".
- Các so sánh trên nhiều bộ dữ liệu chưa hiệu chỉnh so sánh bội, nên chỉ mang tính thăm dò.
