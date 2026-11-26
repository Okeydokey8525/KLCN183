# Kế hoạch đánh giá Polyp-PVT và chốt phần Mamba

Lập ngày 16/11/2026. Bảo vệ ngày 27/11/2026. Mọi kết quả mới **ghi trong thư mục `Baseline_PVT_VMUNet`**. Từ `Test_Mau` chỉ đọc dữ liệu và bảng cũ, không ghi gì vào đó.

## 1. Tình trạng

| Mục | Trạng thái |
|---|---|
| Polyp-PVT 640, seed 0–4 | **Xong**. Mỗi seed đủ 100 epoch, nằm ở `ket_qua/PolypPVT_640/PolypPVT_640_aug_s{0..4}/` |
| Val Dice tốt nhất do notebook tính | 0,9147 · 0,9137 · 0,8986 · 0,9152 · 0,9222 → 0,9129 ± 0,0087 (PraNet cùng seed: 0,9071 ± 0,0042) |
| VM-UNet | **Xong 17/11** (batch 4 × cộng dồn 2, vì batch 8 hết bộ nhớ T4). Dice pixel 0,739 ± 0,025: thấp hơn YOLO, TSVM, PraNet, Polyp-PVT (0/5 seed), chỉ cao hơn U-Net. Xem [ket_qua/VMUNet_640/KET_QUA.md](ket_qua/VMUNet_640/KET_QUA.md) |
| Đánh giá theo pixel như Bảng 4.8 | **Xong** cho cả Polyp-PVT và VM-UNet. So sánh 6 mô hình: [ket_qua/so_sanh_6_mo_hinh/KET_QUA.md](ket_qua/so_sanh_6_mo_hinh/KET_QUA.md) |

Val Dice của notebook tính trên cả 160 ảnh, ảnh nền không báo gì được tính là đúng. Vì vậy **không so trực tiếp được với Bảng 4.8**. Phải chạy bước đánh giá ở mục 3.

## 2. Phần Mamba: có mô hình nào khác để so không?

**Lý do không thay VM-UNet bằng mô hình Mamba khác:**
- Mô hình mới phải chạy cùng số đo với PraNet/U-Net/Polyp-PVT: 640, 100 epoch, batch 8, 5 seed. Với Polyp-PVT, việc đó tốn khoảng 14 giờ T4 cho mỗi seed.
- Giờ GPU Kaggle của các tài khoản tuần này đã dùng gần hết. Đến khi được cấp lại giờ chỉ còn khoảng 5–6 ngày trước buổi bảo vệ, không đủ để chuyển mã sang notebook, chạy thử, chạy 5 seed rồi đánh giá.
- Phần lớn mô hình Mamba cho ảnh y tế dùng chung kernel `mamba_ssm` và kiến trúc VMamba giống VM-UNet, nên ở 640 rất có thể cũng hết bộ nhớ.

| Ứng viên | Ghi chú | Đánh giá |
|---|---|---|
| UltraLight VM-UNet ([github](https://github.com/wurenkai/UltraLight-VM-UNet)) | Rất nhẹ (0,049 M tham số), chắc chắn đủ bộ nhớ ở 640 | Thiết kế cho ảnh da, không phải polyp. Mô hình quá nhỏ, so với nó không có ý nghĩa |
| VMDU-Net ([bài báo](https://www.frontiersin.org/articles/10.3389/frai.2025.1557508/full), [github](https://github.com/sulayman-lee0212/VMDUNet)) | Làm cho polyp, báo Dice 0,934 trên Kvasir-SEG | Lai Mamba với Transformer, nặng hơn VM-UNet. Mã chưa được kiểm chứng |
| PolyMamba, HGM (MICCAI 2025), Polyp-Mamba (MICCAI 2024) | Làm cho polyp | Chưa kiểm tra mã và yêu cầu bộ nhớ. Không kịp thời gian |

**Cập nhật 16/11: nếu vẫn muốn có Mamba thuần thì chạy lại VM-UNet, batch 4 × cộng dồn 2.**
- Notebook `notebooks/vm-unet-640.ipynb` đã có sẵn tùy chọn này: đặt `BATCH_SIZE = 4`, `ACCUM = 2`.
- VM-UNet chỉ dùng LayerNorm, không có BatchNorm (đã kiểm tra mã `models/vmunet/vmamba.py`). Vì vậy cộng dồn 2 bước batch 4 cho gradient gần như bằng batch 8: vẫn **cùng giao thức** với PraNet và Polyp-PVT. Khác biệt duy nhất là drop_path lấy ngẫu nhiên theo từng mini-batch.
- Đây là mô hình Mamba duy nhất đã có mã, trọng số và notebook sẵn sàng. Mô hình khác phải làm lại từ đầu.
- Trước tiên chạy thử 1–2 epoch (`CHAY_THU = True`) để biết số phút/epoch và VRAM. Nếu một seed quá khoảng 20 giờ T4 thì không kịp chạy 5 seed trước 27/11, khi đó dừng và giữ phương án dưới.

**Kết quả chạy thử 16/11 (Kaggle T4, chạy tương tác, batch 4 × 2, 2 epoch):**

| | Giá trị |
|---|---|
| Nạp trọng số VMamba-S | Encoder 117 tensor, decoder 91 tensor, mô hình có 222 tensor |
| Số tham số | 27.427.561 (27,4 M) |
| VRAM huấn luyện | 12,1 GB, vừa T4 (khoảng 15 GB dùng được) |
| Thời gian | 8,7 phút/epoch, nên 100 epoch ≈ **14,5 giờ/seed** (Polyp-PVT: 13,9 giờ) |
| Val Dice sau 2 epoch | 0,0044 → 0,3996. Mô hình đang học, chưa có giá trị để so |

Suy ra:
- **Chạy được cùng giao thức.** 5 seed tốn khoảng 72 giờ GPU T4.
- Mỗi seed cần 2 phiên, vì notebook tự lưu `resume.pth` ở mốc 11 giờ. Mỗi tài khoản cần còn khoảng 15 giờ GPU.
- Chạy song song 5 tài khoản thì xong trong khoảng 1–1,5 ngày.
- **Không đánh giá trên laptop được.** `mamba_ssm` không cài được trên Windows. Phải đánh giá theo pixel bằng một notebook Kaggle riêng: nạp 5 checkpoint, tính chỉ số từng ảnh như Bảng 4.8, rồi tải CSV về ghép với 5 mô hình kia. Độ trễ cũng chỉ đo được trên T4, không so trực tiếp với số đo trên laptop.

**Phương án dự phòng (nếu không chạy VM-UNet):** không thêm mô hình Mamba nào nữa. Cách trả lời hội đồng:
- Đại diện cho hướng Mamba trong khóa luận chính là **YOLO26s-seg + TSVM**: nhánh VMamba được gắn vào YOLO, so ghép cặp 20 seed.
- Đã thử VM-UNet (Mamba thuần, mã gốc của tác giả) ở cùng giao thức 640. Mô hình hết bộ nhớ trên GPU T4 16 GB, nên ghi vào phần giới hạn, kèm con số 14,5 GB đã đo.
- Phía Transformer có Polyp-PVT, chạy cùng giao thức với 5 seed.

## 3. Việc cần làm

> Cập nhật 16/11:
> - Bước 1–5 đã xong. Kết quả ở [ket_qua/danh_gia_pixel/KET_QUA.md](ket_qua/danh_gia_pixel/KET_QUA.md), hình ở `ket_qua/danh_gia_pixel/hinh/`. Tạm chưa đưa lên Canva.
> - Đã đo thêm chi phí của 5 mô hình (tham số, FLOPs, độ trễ, thời gian huấn luyện): [ket_qua/chi_phi/KET_QUA.md](ket_qua/chi_phi/KET_QUA.md).
> - Đang cân nhắc chạy lại VM-UNet với batch 4 × cộng dồn 2. Xem mục 2.

| # | Việc | Lệnh / file | Thời gian | Ai làm |
|---|---|---|---|---|
| 1 | Cài `timm` vào venv GPU, vì `lib/pvtv2.py` của Polyp-PVT cần gói này | `C:\venv_kl_gpu\Scripts\python.exe -m pip install timm` | 1 phút | Claude, khi được duyệt |
| 2 | Chạy thử 1 seed và đối chiếu với Val Dice của notebook (seed 0: 0,9147) để biết đã nạp đúng mô hình | `python danh_gia\danh_gia_pixel_pvt.py --seeds 0 --tag _thu` | ~2 phút | Claude |
| 3 | Chạy đủ 5 seed | `python danh_gia\danh_gia_pixel_pvt.py` | ~10 phút | Claude |
| 4 | Viết bảng kết quả và so ghép cặp vào `ket_qua/danh_gia_pixel/KET_QUA.md` | đọc `12_pixel_thong_ke_5model.csv` | | Claude |
| 5 | Vẽ hình cho slide: cột Dice/IoU của 5 mô hình (seed 0–4), Dice từng seed, ảnh minh họa seed 0 | script mới trong `danh_gia/` | | Claude, gửi xem trước |
| 6 | Đưa vào slide Canva và chuẩn bị câu trả lời | | | Cùng làm, duyệt từng trang |

Script `danh_gia/danh_gia_pixel_pvt.py`:
- Giao thức chép đúng từ `danh_gia_pixel.py` của báo cáo: nhãn polygon ở độ phân giải gốc, ngưỡng 0,5, đủ 120 ảnh polyp và 40 ảnh nền, có thêm nhãn phụ Kvasir-SEG.
- Dự đoán bằng `sigmoid(P1 + P2)`, đúng như notebook dùng để chọn checkpoint.
- Lấy số từng ảnh của YOLO, TSVM, PraNet, U-Net seed 0–4 từ `10_pixel_tung_anh.csv` có sẵn, nên không phải suy luận lại 4 mô hình đó.
- So ghép cặp Polyp-PVT với TSVM, YOLO và PraNet: Δ, paired t, Wilcoxon, số seed Polyp-PVT cao hơn.

Kết quả ghi vào `ket_qua/danh_gia_pixel/`:

| File | Nội dung |
|---|---|
| `10_pixel_tung_anh_polyppvt.csv` | Polyp-PVT × 5 seed × 160 ảnh |
| `11_pixel_tung_seed_5model.csv` | 5 mô hình × seed 0–4 |
| `12_pixel_thong_ke_5model.csv` | mean ± SD và so ghép cặp |
| `du_lieu_dinh_tinh/pred_seed0_polyppvt640.npz` | mặt nạ seed 0 để vẽ ảnh minh họa |

## 4. Lưu ý khi trình bày

- Chỉ có 5 seed, nên Wilcoxon thấp nhất chỉ được p = 0,0625. Báo mean ± SD, paired t và số seed thắng (ví dụ 4/5). Viết "xu hướng", không viết "vượt trội".
- So TSVM với Polyp-PVT trên **seed 0–4**, không lấy 20 seed. Nếu bảng chính dùng 20 seed thì ghi rõ hai cách lấy seed khác nhau.
- Seed 2 của Polyp-PVT đạt tốt nhất ngay từ epoch 29 (0,8986). Đây vẫn là kết quả hợp lệ, phải giữ, không chạy lại.
- Khác bản gốc Polyp-PVT: ảnh 640 thay 352, batch 8 thay 16, không huấn luyện đa tỉ lệ, tăng cường giống YOLO. Không so với con số trong bài báo gốc.
- Chọn checkpoint trên chính tập val dùng để báo cáo, giống 4 mô hình còn lại.

## 5. Có thể làm thêm (tùy thời gian)

- Đánh giá Polyp-PVT trên 800 ảnh manh tràng (báo nhầm ảnh nền) và trên dữ liệu ngoài. Hai bộ này đã có trong báo cáo cho 4 mô hình kia. Mỗi bộ cần một script riêng trong `danh_gia/`, chép giao thức từ `Test_Mau/08_du_lieu_thuc_te/scripts/`.
- Đo số tham số, FLOPs và độ trễ của Polyp-PVT trên laptop, để so chi phí với TSVM.
