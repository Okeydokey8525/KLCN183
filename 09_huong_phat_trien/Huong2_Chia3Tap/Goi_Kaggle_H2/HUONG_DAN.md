# Hướng 2: chạy YOLO26s-seg và TSVM, 10 seed, trên Kaggle

Tạo ngày 17/11/2026. Dữ liệu là BG20 cũ chia lại thành 3 tập (xem [`../KE_HOACH.md`](../KE_HOACH.md)). Siêu tham số giữ nguyên như 40 lượt trong báo cáo.

## Trong thư mục này

| File | Dùng để |
|---|---|
| `1_dataset/kvasir-bg20-3tap-tsvm.zip` (100 MB) | Một dataset gồm `Kvasir_BG20_3tap/` (train/val/test) và `ultralytics_TSVM_FP32/` (mã Ultralytics 8.4.127 + TSVM, lưu FP32) |
| `3_notebooks/h2-chay-thu-tsvm.ipynb`, `h2-chay-thu-yolo.ipynb` | Chạy thử 1 epoch, mỗi file khoảng 10 phút |
| `3_notebooks/h2-yolo-s0-3.ipynb` … `h2-tsvm-s8-9.ipynb` | 8 notebook chạy thật, đã đặt sẵn `MODEL` và `SEEDS` |

## Bước 1. Tạo dataset (tài khoản chính `okeydokey0805`)

1. **Create → New Dataset**, kéo `kvasir-bg20-3tap-tsvm.zip` vào, đặt tên `kvasir-bg20-3tap-tsvm`, để **Private**, bấm Create. Không cần giải nén.
2. Vào **Settings → Sharing** của dataset, thêm 5 tài khoản còn lại với quyền **Can view**, rồi **Save**. Làm giống lần chia sẻ dataset VM-UNet.

## Bước 2. Chạy thử (bắt buộc, trước khi chạy thật)

Ở một tài khoản bất kỳ:
1. **Create → New Notebook → File → Import Notebook**, chọn `h2-chay-thu-tsvm.ipynb`.
2. **Add Input**: dán `https://www.kaggle.com/datasets/okeydokey0805/kvasir-bg20-3tap-tsvm` vào ô "Search keyword or URL", bấm ⊕.
3. **Settings → Accelerator → GPU T4 x2**, **Internet → On** (để tải `yolo26s-seg.pt`).
4. **Run All** (chạy tương tác cũng được, chỉ khoảng 10 phút).
5. Làm lại với `h2-chay-thu-yolo.ipynb`.

Log phải có:
- `✅ Ultralytics 8.4.127 (bản TSVM_FP32)` và `📁 Dữ liệu: ... {'train': 880, 'val': 160, 'test': 160}`
- TSVM: `Tầng 10: C2TSVMamba, 1,811,201 tham số`. YOLO: `Tầng 10: C2PSA, 990,976 tham số`
- `✅ seed 0: test mask mAP50-95 ...` và dòng `Xong phiên.`

**Gửi mình mấy dòng này (hoặc ảnh chụp) để kiểm tra trước khi chạy thật.** Tab Output phải có `runs_H2_BG20_3tap_..._s0_chaythu.zip`.

## Bước 3. Chạy thật: 8 phiên

Mỗi phiên dùng 1 notebook. Làm như Bước 2 (Import, Add Input, T4, Internet On), nhưng bấm **Save Version → Save & Run All (Commit)** để chạy nền.

| Notebook | Seed | Ước tính |
|---|---|---|
| `h2-yolo-s0-3` | YOLO 0, 1, 2, 3 | ~8–9 giờ |
| `h2-yolo-s4-7` | YOLO 4, 5, 6, 7 | ~8–9 giờ |
| `h2-yolo-s8-9` | YOLO 8, 9 | ~4–5 giờ |
| `h2-tsvm-s0-1` | TSVM 0, 1 | ~8–9 giờ |
| `h2-tsvm-s2-3` | TSVM 2, 3 | ~8–9 giờ |
| `h2-tsvm-s4-5` | TSVM 4, 5 | ~8–9 giờ |
| `h2-tsvm-s6-7` | TSVM 6, 7 | ~8–9 giờ |
| `h2-tsvm-s8-9` | TSVM 8, 9 | ~8–9 giờ |

Tổng khoảng 65–70 giờ GPU. Gợi ý chia 6 tài khoản: 6 phiên đầu mỗi tài khoản một phiên chạy song song. Hai tài khoản xong sớm nhất chạy thêm `h2-yolo-s8-9` và phiên còn lại. Mỗi tài khoản cần còn khoảng 10 giờ GPU cho mỗi phiên.

Ước tính lấy từ log báo cáo (YOLO 2,3 giờ/seed, TSVM 4,6 giờ/seed). Lần này train ít ảnh hơn (880 thay 1.040), nên có thể nhanh hơn một chút.

**Tự dừng an toàn:** trước mỗi seed, notebook kiểm tra thời gian đã chạy cộng thời gian của seed trước. Nếu vượt 11 giờ thì **bỏ qua** seed đó và in `⏸ Bỏ qua seed ...`. Seed bị bỏ qua thì báo mình, mình làm notebook riêng cho nó. Mỗi seed chạy xong được nén ngay thành zip riêng, nên phiên có bị cắt thì các seed đã xong vẫn còn.

## Bước 4. Tải kết quả

Ở tab **Output** của mỗi phiên, tải các file `runs_H2_BG20_3tap_<YOLO26s_seg|TSVM_FP32>_s<N>.zip` và để chung một chỗ (ví dụ Downloads). Mình sẽ giải nén vào `Huong2_Chia3Tap/ket_qua/`.

Mỗi zip gồm `results.csv`, `args.yaml`, `weights/best.pt`, `last.pt` và `test_metrics.json`. File `test_metrics.json` chứa mAP trên tập test, tính một lần bằng `best.pt`.

## Lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `❌ Chưa thêm dataset chứa ...` | Thiếu Input `kvasir-bg20-3tap-tsvm` |
| `❌ Tải chưa đủ` | Bật Settings → Internet → On rồi chạy lại |
| `AssertionError` ở số ảnh | Dataset bị giải nén sai. Báo mình |
| Queued lâu | Chạy tương tác (Edit → Run All), để trang mở. Xong thì Save Version → Quick Save để giữ Output |

## Lưu ý

- Không đổi `SEEDS`, không chạy seed ngoài 0–9, không chạy lại seed đã có kết quả.
- Chỉ chọn checkpoint trên val. Tập test chỉ được đánh giá một lần sau cùng (notebook tự làm).
- Không để dataset Public.
