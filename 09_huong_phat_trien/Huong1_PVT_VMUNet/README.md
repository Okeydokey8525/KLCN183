# Huấn luyện thêm Polyp-PVT và VM-UNet (hướng 1, sau khi nộp)

Mục đích: trả lời ý II.4 của GVHD bằng cách so sánh thêm với một mô hình Transformer (Polyp-PVT) và một mô hình Mamba (VM-UNet) chuyên cho polyp. Kết quả chỉ dùng cho slide và phần chuẩn bị trả lời hội đồng, không sửa bản đã nộp.

Thư mục này độc lập: không import file Python nào trong `Test_Mau`. Phần dữ liệu, tăng cường và vòng huấn luyện được **chép** từ `pranet-new.ipynb`, nên cả hai mô hình dùng cùng giao thức với PraNet/U-Net 640.

## Cấu trúc

| Đường dẫn | Nội dung |
|---|---|
| `tao_notebook.py` | Sinh 3 notebook. Muốn sửa thì sửa file này rồi chạy `python tao_notebook.py`, không sửa tay trong `.ipynb` |
| `notebooks/polyp-pvt-640.ipynb` | Huấn luyện Polyp-PVT |
| `notebooks/vm-unet-640.ipynb` | Huấn luyện VM-UNet |
| `notebooks/dong-goi-mamba-wheel.ipynb` | Chạy **một lần** để đóng gói kernel `mamba_ssm` cho VM-UNet |
| `KE_HOACH_DANH_GIA.md` | Kế hoạch và tình trạng đánh giá hướng 1 |
| `goi_kaggle/PVT/`, `goi_kaggle/VMUNet/` | Gói đưa lên Kaggle (dataset zip, notebook từng seed, `HUONG_DAN.md`) |
| `ma_nguon_goc/Polyp-PVT-main/`, `ma_nguon_goc/VM-UNet-main/` | Mã gốc của tác giả, dùng khi đánh giá trên laptop |
| `danh_gia/` | Script đánh giá theo pixel, đo chi phí, vẽ hình |
| `ket_qua/PolypPVT_640/`, `ket_qua/VMUNet_640/` | Kết quả 5 seed mỗi mô hình (đã giải nén từ `runs_*.zip`) |
| `ket_qua/danh_gia_pixel/`, `ket_qua/chi_phi/`, `ket_qua/so_sanh_6_mo_hinh/` | Bảng, hình và `KET_QUA.md` |

Trọng số tải về (`pvt_v2_b2.pth`, VMamba, wheel `mamba_ssm`) nằm chung ở `../trong_so_tai_ve/`. Sắp xếp lại ngày 18/11/2026. Các bước bên dưới ghi lại cách đã làm lúc đầu, nên đường dẫn `ma_nguon_goc/polyp-pvt-code/`… là đường dẫn cũ.

## Giao thức (giống PraNet/U-Net 640)

| Mục | Giá trị |
|---|---|
| Dữ liệu | BG20: train 880 polyp + 160 nền, val 120 polyp + 40 nền |
| Ảnh | 640×640, kéo thẳng về hình vuông, chuẩn hóa ImageNet |
| Tăng cường | Giống YOLO: HSV (0,015/0,7/0,4), lật ngang 0,5, dịch 0,1, co giãn 0,5, mosaic tắt ở 10 epoch cuối |
| Epoch, batch | 100 epoch, batch 8 (hoặc 4 × cộng dồn 2 nếu thiếu bộ nhớ) |
| Seed | **0, 1, 2, 3, 4**, chốt trước, không thêm seed sau khi xem kết quả |
| Chọn checkpoint | Dice val cao nhất (ảnh nền tính đúng nếu không báo điểm nào) |

Thiết lập riêng, lấy từ repo gốc:

| | Polyp-PVT | VM-UNet |
|---|---|---|
| Mã nguồn | github.com/DengPingFan/Polyp-PVT (chỉ dùng cho nghiên cứu, giáo dục) | github.com/JCruan519/VM-UNet (Apache-2.0) |
| Backbone | PVTv2-B2, `pvt_v2_b2.pth` | VMamba-S, `vmamba_small_e238_ema.pth`, nạp cho cả encoder và decoder |
| Hàm mất mát | structure loss (BCE + IoU có trọng số) cho P1 và P2 | BCE + Dice |
| Bộ tối ưu | AdamW lr 1e-4, wd 1e-4, lr không đổi | AdamW lr 1e-3, wd 1e-2, cosine T_max 50, eta_min 1e-5 |
| Khác | Cắt gradient từng phần tử ±0,5; dự đoán = sigmoid(P1 + P2) | depths [2,2,2,2], decoder [2,2,2,1], drop_path 0,2 |
| Khác bản gốc | Bỏ huấn luyện đa tỉ lệ; 640 thay 352; batch 8 thay 16 | 640 thay 256; batch 8 thay 32; 100 epoch thay 300; tăng cường YOLO thay lật dọc/xoay |

## Việc cần làm trên Kaggle

### Bước 1. Tải về máy (bạn làm)

| Thứ cần tải | Lấy ở đâu | Để vào |
|---|---|---|
| Mã Polyp-PVT | github.com/DengPingFan/Polyp-PVT → nút **Code → Download ZIP** | `ma_nguon_goc/polyp-pvt-code/` |
| `pvt_v2_b2.pth` | Google Drive trong README của Polyp-PVT, mục "pretrained model" | `ma_nguon_goc/polyp-pvt-code/` |
| Mã VM-UNet | github.com/JCruan519/VM-UNet → **Code → Download ZIP** | `ma_nguon_goc/vm-unet-code/` |
| `vmamba_small_e238_ema.pth` | Google Drive trong README của VM-UNet, mục "pre-trained weights" | `ma_nguon_goc/vm-unet-code/` |

Không cần giải nén ZIP mã nguồn: Kaggle tự giải nén khi tải lên. Notebook tự dò file `lib/pvt.py`, `models/vmunet/vmunet.py` và hai file `.pth` ở bất kỳ đâu trong `/kaggle/input`.

### Bước 2. Tạo 2 Kaggle Dataset (bạn làm, ở tài khoản chính)

1. `polyp-pvt-code`: tải lên ZIP Polyp-PVT và `pvt_v2_b2.pth`.
2. `vm-unet-code`: tải lên ZIP VM-UNet và `vmamba_small_e238_ema.pth`.
3. Để chế độ **Private**, rồi vào **Settings → Sharing** của từng dataset, thêm 5 tài khoản còn lại. Không để Public, vì Polyp-PVT không cho phân phối lại tự do.
4. Kiểm tra 5 tài khoản kia mở được dataset BG20 `okeydokey0805/kvasir-yolo-seg-bg20`. Nếu dataset này đang Private thì chia sẻ thêm.

### Bước 3. Đóng gói kernel Mamba (chạy một lần)

1. Tạo notebook mới từ `dong-goi-mamba-wheel.ipynb`. Bật **GPU T4** và **Internet** (tài khoản phải xác minh số điện thoại).
2. Chạy. Có thể mất 30–60 phút nếu phải biên dịch.
3. Tải file `wheels/mamba_ssm-*.whl` ở tab Output, rồi thêm vào dataset `vm-unet-code` (New Version).

Nếu bước này lỗi, gửi log cho mình. Không có wheel thì notebook VM-UNet vẫn tự cài từ PyPI khi bật Internet, nhưng mỗi lượt mất thêm thời gian biên dịch.

### Bước 4. Chạy thử (mỗi mô hình một lần)

1. Import notebook. Add Input: BG20 và dataset mã tương ứng. Bật GPU T4 (VM-UNet bật thêm Internet nếu chưa có wheel).
2. Ở Cell 0 đặt `CHAY_THU = True`, rồi **Run All**.
3. Gửi mình dòng `≈ ... phút/epoch` và cột `gpu_mem_gb`. Nếu báo hết bộ nhớ (CUDA out of memory), đổi `BATCH_SIZE = 4`, `ACCUM = 2` rồi chạy lại.

Bước này kiểm tra: mã gốc nạp được, backbone nhận trọng số ImageNet (notebook in số tensor khớp và dừng nếu không khớp), kernel Mamba chạy được.

### Bước 5. Chạy chính

Mỗi lượt: đặt `CHAY_THU = False`, sửa `SEED`, rồi bấm **Save Version → Save & Run All (Commit)** để chạy nền.

| Tài khoản | Lượt |
|---|---|
| 1 | Polyp-PVT seed 0, rồi seed 1 |
| 2 | Polyp-PVT seed 2, rồi seed 3 |
| 3 | Polyp-PVT seed 4, rồi VM-UNet seed 4 |
| 4 | VM-UNet seed 0, rồi seed 1 |
| 5 | VM-UNet seed 2, rồi seed 3 |
| 6 | Dự phòng chạy lại lượt hỏng |

Nếu một lượt chạy quá 11 giờ, notebook tự lưu `resume.pth` rồi dừng (Kaggle cắt phiên ở 12 giờ). Để chạy tiếp: thêm Output của lượt đó làm Input, giữ nguyên `SEED`, rồi chạy lại. Notebook tự tìm `resume.pth` và chạy tiếp từ epoch kế.

### Bước 6. Gửi kết quả về

Tải `runs_<Tên>_640_aug_s<SEED>.zip` ở tab Output, giải nén vào `ket_qua/<Tên>_640/`. Mỗi lượt gồm:
- `results.csv`: loss, Dice/IoU val, lr, mosaic, thời gian, bộ nhớ GPU theo epoch;
- checkpoint tốt nhất `polyp_pvt_best_s*.pth` hoặc `vmunet_best_s*.pth`, và `last.pth`;
- `tom_tat.txt`: epoch tốt nhất, thời gian, phiên bản thư viện.

Sau đó mình viết script đánh giá theo pixel (cùng chỉ số với PraNet/U-Net 640) ngay trong thư mục này, và so ghép cặp seed 0–4.

## Lưu ý khi báo cáo

- Với 5 seed, Wilcoxon hai phía có p nhỏ nhất là 0,0625, nên không bao giờ đạt mức ý nghĩa 0,05. Báo trung bình ± SD, paired t và số seed thắng.
- Checkpoint được chọn trên tập val dùng để báo cáo (giống mọi mô hình khác của đề tài), nên vẫn có thiên lệch lựa chọn.
- Đã kiểm trên máy (15/11/2026): phần dữ liệu, vòng huấn luyện, dừng-lưu-chạy tiếp, hai hàm mất mát, chạy với mô hình giả. **Chưa** chạy được mô hình thật, vì máy chưa có mã gốc, trọng số và `mamba_ssm` (Windows không cài được). Bước 4 trên Kaggle là lần kiểm đầu tiên với mô hình thật.
