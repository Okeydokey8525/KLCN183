# Hướng 4: SS2D quét bằng kernel CUDA (tăng tốc TSVM, giữ nguyên mô hình)

Tạo ngày 24/11/2026. Đây là hướng phát triển làm sau khi nộp khóa luận, không thay đổi bản đã nộp.

## Mục tiêu

Trong bản của báo cáo (`Test_Mau/02_ma_nguon/ultralytics_TSVM_FP32`), SS2D quét bằng PyTorch thuần theo kiểu song song Hillis–Steele:
- 4 hướng chạy lần lượt;
- mỗi hướng lặp 9 vòng trên dãy L = 400 (đệm lên 512).

Theo ablation 20 seed, phần chậm thêm của TSVM gần như hoàn toàn do SS2D:

| Cấu hình | Giờ huấn luyện |
|---|---|
| Baseline | 2,32 |
| Chỉ nhánh hình dạng | 2,37 |
| Chỉ SS2D | 4,64 |
| TSVM đầy đủ | 4,68 |

Hướng 4 thay phép quét bằng kernel `selective_scan_fn` của `mamba_ssm`. Phương trình, tham số và tên trọng số giữ nguyên. Mục tiêu là đo xem thời gian huấn luyện, độ trễ và bộ nhớ giảm được bao nhiêu, trong khi mAP không đổi.

**Dữ liệu:** BG20 của báo cáo, train 1.040 / val 160 (`Test_Mau/01_du_lieu/Kvasir_YOLO_SEG_BG20`).

**Siêu tham số:** như báo cáo (AdamW, lr0 0,001, batch 8, 100 epoch, `amp=False`, deterministic, close_mosaic 10, warmup 5).

**Mốc so sánh** (TSVM bản PyTorch, seed 0):

| | Giá trị |
|---|---|
| val mask mAP50-95 tại epoch tốt nhất (88) | 0,7245 |
| Giờ huấn luyện | 4,52 |

20 seed: trung bình 0,722, SD 0,0074, 4,68 giờ (4,52–4,96). Độ trễ forward trên RTX 5050: 39,86 ms (baseline 26,33 ms).

**Quy tắc sau khi huấn luyện seed 0 bằng kernel:**
- |Δ mAP50-95| ≤ 0,0074 (1 SD): coi như tương đương, dừng ở 1 seed.
- Lớn hơn 0,0074: chạy thêm seed 1–4.

## Cấu trúc

```
Huong4_KernelCUDA/
  README.md
  ma_nguon/ultralytics_TSVM_CUDA/   bản chép TSVM_FP32; chỉ khác nn/modules/topology_shape_vmamba.py
  kiem_tra/kiem_tra_moc1.py         test tương đương trên laptop
  kiem_tra/ket_qua_moc1.txt         log lần chạy 24/11
```

## Thay đổi trong mã

Tệp `ma_nguon/ultralytics_TSVM_CUDA/nn/modules/topology_shape_vmamba.py`:

- **`SS2D.forward`** chọn cách quét theo biến môi trường `TSVM_SCAN`:
  - `auto` (mặc định): có `mamba_ssm` và tensor đang ở trên GPU thì dùng kernel, ngược lại dùng PyTorch.
  - `pytorch`: luôn dùng bản cũ.
  - `cuda`: bắt buộc dùng kernel, báo lỗi nếu không có.
- **`SS2D._forward_pytorch`** là mã cũ, giữ nguyên từng dòng.
- **`SS2D._forward_cuda`** xếp 4 hướng thành một lần gọi kernel:
  - `u`, `delta`: (B, 4D, L), với `delta` là đầu ra `dt_projs` trước softplus và gọi kernel với `delta_softplus=True`;
  - `A`: (4D, N);
  - `B`, `C`: (B, 4, N, L);
  - `D` và `z` không dùng (cổng SiLU(z) nhân sau khi cộng 4 hướng, như cũ).
- **Khác biệt số học duy nhất:** bản PyTorch kẹp Δ·A trong [−50, 0] trước exp, còn kernel thì không. Hai bản chỉ khác khi exp(Δ·A) < 2·10⁻²², nên sai khác không đáng kể.

Máy không có `mamba_ssm` (laptop Windows, CPU, web PolypVision) tự dùng bản PyTorch. `best.pt` của bản này nạp được vào bản kia.

## Mốc 1 (24/11): kiểm tra trên laptop, chưa có kernel thật. Tất cả PASS

Lệnh chạy: `C:\venv_kl_gpu\Scripts\python.exe -X utf8 kiem_tra/kiem_tra_moc1.py`, mất khoảng 1 phút GPU. Log ở `kiem_tra/ket_qua_moc1.txt`.

| Test | Nội dung | Kết quả |
|---|---|---|
| T1 | So 369 tệp với bản báo cáo | Chỉ `topology_shape_vmamba.py` khác |
| T2 | Nhánh PyTorch so với bản gốc, trên CPU và GPU (đầu ra, grad đầu vào, grad tham số; SS2D và C2TSVMamba) | Trùng bit 100% |
| T3 | Hillis–Steele so với quét tuần tự, float64, L = 1…513 | Sai khác ≤ 1,8·10⁻¹⁵ |
| T4 | `_forward_cuda` chạy với một `selective_scan` viết tay theo đúng quy ước `mamba_ssm`, so với nhánh PyTorch (float64, forward và backward) | Sai khác tương đối ≤ 1,2·10⁻¹⁵ (cách xếp 4 hướng là đúng) |
| T5 | `TSVM_SCAN=cuda` khi không có `mamba_ssm`; `auto` | Báo lỗi rõ; `auto` tự dùng PyTorch |
| T6 | `best.pt` seed 0 của báo cáo, đánh giá 160 ảnh val bằng hai nhánh | Cả hai đều mask mAP50-95 0,7238, mAP50 0,9026, box 0,7138 (chênh 0). Tầng 10 có 1.811.201 tham số, tổng 12.254.688 |

**Ghi chú T6:** 0,7238 khác 0,7245 trong `results.csv`. Lý do: `results.csv` là giá trị đánh giá trong lúc huấn luyện (mô hình EMA, chưa gộp Conv-BN), còn T6 đánh giá độc lập trên mô hình đã gộp Conv-BN. Hai nhánh quét cho kết quả như nhau.

**Chưa kiểm được ở mốc 1:** kernel thật. Laptop không có `mamba_ssm`. Thư mục `trong_so_tai_ve/mamba_wheels` chỉ có wheel cho torch 1.13 / Python 3.8 (dùng cho VM-UNet ở hướng 1), không dùng được với Ultralytics 8.4.127.

## Mốc 2 (24/11): đo kernel trên Kaggle, xong

Xem [`Goi_Kaggle_H4/HUONG_DAN.md`](Goi_Kaggle_H4/HUONG_DAN.md).

**Gói gồm:**
- `1_dataset/h4-tsvm-cuda.zip` (176 MB): mã, dữ liệu BG20, `best.pt` seed 0 của baseline và TSVM, tệp `h4_kiem_tra.py`.
- Notebook `h4-moc2-kiem-tra-kernel.ipynb`. Tạo lại gói bằng `scripts/tao_goi_kaggle_h4.py`.

**Notebook làm gì:**
- Cài `mamba-ssm` và giữ lại wheel để dùng cho mốc 3.
- Chạy 6 phần A–F:

| Phần | Nội dung |
|---|---|
| A | So kernel thật với PyTorch |
| B | Đánh giá val 160 ảnh |
| C | Độ trễ forward |
| D | Độ trễ riêng khối SS2D |
| E | Thời gian một bước huấn luyện và bộ nhớ GPU |
| F | Giả lập video 160 khung |

**Đã chạy thử toàn bộ A–F trên laptop** bằng `kiem_tra/chay_thu_moc2_laptop.py`, dùng kernel giả lập và số lần đo nhỏ. Mã không lỗi, phần A và B đều PASS. Số đo tốc độ trên laptop không có ý nghĩa, vì kernel giả lập là một vòng lặp Python.

### Kết quả mốc 2 trên Kaggle (24/11, Tesla T4, torch 2.11.0+cu130, Python 3.13, mamba_ssm 2.3.2.post1 tự biên dịch 50 phút)

Số liệu gốc ở `ket_qua_moc2_kaggle/`. Wheel giữ ở `Downloads/mamba_ssm-2.3.2.post1-cp313-cp313-linux_x86_64.whl` (349 MB).

| Phép đo | Baseline | TSVM PyTorch | TSVM kernel |
|---|---|---|---|
| A: sai khác kernel so với PyTorch (đầu ra / grad) | | | ≤ 4,2·10⁻⁷ / ≤ 4,5·10⁻⁶ |
| B: mask mAP50-95 val 160 | 0,7312 | 0,7242 | 0,7242 (trùng) |
| C: forward batch 1 (trung vị ms) | 19,24 | 37,52 (1,95×) | 19,86 (1,03×) |
| D: riêng SS2D, batch 1 / batch 8 (ms) | | 18,9 / 149,2 | 1,75 / 2,99 |
| E: 1 bước huấn luyện batch 8 (ms), bộ nhớ đỉnh | 498, 5,49 GB | 1074, 10,92 GB | 513, 5,62 GB |
| F: video 160 khung (FPS) | 39,5 | 22,8 | 38,2 |

Ghi chú: lỗi "không cài được" lần đầu là do phép kiểm tra import `selective_scan_cuda` trước `torch` (thiếu `libc10.so`); đã sửa trong script.

## Mốc 3 (24–26/11): huấn luyện 20 seed bằng kernel, xong

Xem [`Goi_Kaggle_H4/HUONG_DAN_MOC3.md`](Goi_Kaggle_H4/HUONG_DAN_MOC3.md).

- Notebook `h4-moc3-huan-luyen-s0-s15.ipynb` được tạo bằng `scripts/tao_goi_kaggle_h4_moc3.py`.
- Siêu tham số: 106 tham số chép từ `args.yaml` lượt báo cáo. Đã kiểm tra bằng `get_cfg` của bản fork.
- Cài `mamba_ssm` từ dataset `h4-mamba-wheel`.

### Kết quả mốc 3 (26/11, 20 seed, ghép cặp với bản báo cáo)

Chi tiết ở [`ket_qua_moc3/KET_QUA.md`](ket_qua_moc3/KET_QUA.md).

| | PyTorch | Kernel CUDA | Δ, CI 95%, p |
|---|---|---|---|
| mask mAP50-95 tốt nhất | 0,7232 ± 0,0072 | 0,7212 ± 0,0088 | −0,0020 [−0,0056; +0,0016], p = 0,25 |
| Giờ huấn luyện / seed | 4,68 | 2,46 | −47%, 20/20 seed |

### Đánh giá thêm (26/11)

- **Tập test Kvasir-SEG** (theo giao thức Bảng 4.8): Dice của TSVM CUDA 0,8467, TSVM PyTorch 0,8540, YOLO 0,8486. CUDA − PyTorch = −0,0073 (p = 0,11).
- **Dữ liệu ngoài và 800 ảnh manh tràng** (theo giao thức Mục 4.7): không bộ nào khác với TSVM PyTorch (mọi p ≥ 0,15). Trên CVC-ClinicDB và CVC-ColonDB, TSVM CUDA vẫn thấp hơn YOLO, giống TSVM PyTorch trong báo cáo.
- **Hình:** `ket_qua_moc3/hinh/`, mỗi biểu đồ một tệp (`h4_01a`–`f`, `h4_02a`–`c`).

## Việc tiếp theo

1. Đưa hình vào slide hoặc phụ lục nếu cần.
2. Có thể thử kernel cho phần suy luận trên web (cần Linux có `mamba_ssm`).
