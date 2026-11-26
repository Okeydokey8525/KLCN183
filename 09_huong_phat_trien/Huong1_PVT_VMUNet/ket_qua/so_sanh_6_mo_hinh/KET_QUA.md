# So sánh 6 mô hình (seed 0–4)

YOLO26s-seg, YOLO26s-seg + TSVM, PraNet, Polyp-PVT, VM-UNet, U-Net. Ảnh 640, BG20, đánh giá theo pixel như Bảng 4.8. Lập ngày 17/11/2026.

Thư mục này chỉ chứa **hình gộp**. Các hình và bảng 5 mô hình cũ (`danh_gia_pixel/hinh/`, `chi_phi/hinh/`) giữ nguyên, không ghi đè. Kết quả riêng của từng mô hình mới:
- Polyp-PVT: [`../danh_gia_pixel/KET_QUA.md`](../danh_gia_pixel/KET_QUA.md) và [`../chi_phi/KET_QUA.md`](../chi_phi/KET_QUA.md)
- VM-UNet: [`../VMUNet_640/KET_QUA.md`](../VMUNet_640/KET_QUA.md)

## Bảng tóm tắt

| Mô hình | Kiểu | Dice pixel | Tham số | GFLOPs | Huấn luyện (giờ/seed) | Ảnh nền báo nhầm |
|---|---|---:|---:|---:|---:|---:|
| Polyp-PVT | Transformer | 0,894 ± 0,005 | 25,1 M | 85,0 | 13,9 | 3,0% |
| PraNet | CNN | 0,885 ± 0,005 | 24,6 M | 72,4 | 4,0 | 2,5% |
| YOLO26s-seg | YOLO | 0,854 ± 0,022 | 11,4 M | 34,3 | 2,3 | 0,5% |
| YOLO26s-seg + TSVM | YOLO + Mamba | 0,854 ± 0,011 | 12,3 M | ≥ 34,8 | 4,6 | 2,0% |
| VM-UNet | Mamba | 0,739 ± 0,025 | 27,4 M | ≥ 66,1 | 14,4 | 14,5% |
| U-Net | CNN | 0,697 ± 0,025 | 7,8 M | 150,9 | 5,7 | 62,5% |

Nguồn: [`30_chi_phi_6_mo_hinh.csv`](30_chi_phi_6_mo_hinh.csv), [`31_thoi_gian_huan_luyen_6_mo_hinh.csv`](31_thoi_gian_huan_luyen_6_mo_hinh.csv), [`../danh_gia_pixel/12_pixel_thong_ke_6model.csv`](../danh_gia_pixel/12_pixel_thong_ke_6model.csv).

Bảng không có cột độ trễ vì VM-UNet chưa đo được (lý do ở `VMUNet_640/KET_QUA.md`). Độ trễ của 5 mô hình kia vẫn ở `chi_phi/KET_QUA.md`.

## Hình

Vẽ bằng [`danh_gia/ve_hinh_vmunet.py`](../../danh_gia/ve_hinh_vmunet.py), lưu ở `hinh/` (PNG 300 dpi và PDF):

| File | Nội dung |
|---|---|
| `s01_cot_pixel_6_mo_hinh` | Dice, IoU, Precision, Recall theo pixel của 6 mô hình |
| `s02_dice_tung_seed` | Dice từng seed, đường xám nối cùng seed |
| `s03_minh_hoa_seed0` | Ảnh minh họa seed 0 (cùng 3 ảnh với hình h03 cũ), thêm cột VM-UNet. U-Net không có cột riêng, giống h03 |
| `s04_duong_cong_val` | Val Dice của notebook theo epoch: PraNet, U-Net, Polyp-PVT, VM-UNet (trung bình và min–max 5 seed) |
| `s05_chi_phi_6_mo_hinh` | Tham số, GFLOPs, giờ huấn luyện |
| `s06_dice_va_chi_phi` | Dice theo GFLOPs, cỡ chấm theo số tham số |

## Lưu ý khi trình bày

- 5 seed, nên Wilcoxon thấp nhất chỉ được 0,0625. Báo mean ± SD, paired t và số seed thắng. Viết "xu hướng", không viết "vượt trội".
- PraNet, U-Net, Polyp-PVT, VM-UNet chỉ cho mặt nạ. YOLO và TSVM cho từng polyp riêng (hộp và mặt nạ).
- Thời gian huấn luyện lấy từ log Kaggle. Polyp-PVT và VM-UNet chạy trên T4. Các lượt khác không ghi loại GPU.
