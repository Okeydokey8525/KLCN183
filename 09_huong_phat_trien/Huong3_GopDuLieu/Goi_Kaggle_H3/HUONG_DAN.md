# Hướng 3: chạy YOLO26s-seg và TSVM, 10 seed mỗi mô hình (0–4 xong 22/11, 5–9 đang chạy), trên Kaggle

Tạo ngày 19/11/2026 bằng [`../scripts/tao_goi_kaggle_h3.py`](../scripts/tao_goi_kaggle_h3.py). Dữ liệu là bộ gộp 3 tập (xem [`../README.md`](../README.md)). Siêu tham số giữ nguyên như 40 lượt trong báo cáo và 20 lượt hướng 2.

## Trong thư mục này

| File | Dùng để |
|---|---|
| `1_dataset/h3-gop-3tap-tsvm.zip` (255 MB) | Một dataset gồm `H3_Gop_3tap/` (train 2.455 / val 532 / test 522, ảnh + nhãn) và `ultralytics_TSVM_FP32/` (cùng mã đã chạy hướng 2) |
| `3_notebooks/h3-chay-thu-tsvm.ipynb` | Chạy thử TSVM 1 epoch, khoảng 15 phút |
| `3_notebooks/h3-yolo-s0.ipynb`, `h3-yolo-s1-2.ipynb`, `h3-yolo-s3-4.ipynb` | YOLO: seed 0 riêng, rồi 2 seed mỗi phiên |
| `3_notebooks/h3-tsvm-s0.ipynb` … `h3-tsvm-s4.ipynb` | TSVM: mỗi phiên 1 seed |
| `3_notebooks/h3-yolo-s5.ipynb` … `h3-yolo-s9.ipynb`, `h3-tsvm-s5.ipynb` … `h3-tsvm-s9.ipynb` | Seed 5–9 (thêm 22/11), mỗi phiên 1 seed |
| `3_notebooks/h3-chay-tiep.ipynb` | Chỉ dùng nếu một seed bị dừng ở mốc an toàn (xem cuối file) |

## Bước 1. Tạo dataset (tài khoản chính `okeydokey0805`)

1. **Create → New Dataset**, kéo `h3-gop-3tap-tsvm.zip` vào, đặt tên `h3-gop-3tap-tsvm`, để **Private**, bấm Create. Không cần giải nén.
2. **Settings → Sharing** của dataset: thêm 5 tài khoản còn lại, quyền **Can view**, **Save**.

## Bước 2. Chạy thử (nên làm trước)

1. **Create → New Notebook → File → Import Notebook**, chọn `h3-chay-thu-tsvm.ipynb`.
2. **Add Input**: dán `https://www.kaggle.com/datasets/okeydokey0805/h3-gop-3tap-tsvm`, bấm ⊕.
3. **Settings → Accelerator → GPU T4 x2**, **Internet → On**.
4. **Run All** (chạy tương tác, khoảng 15 phút).

Log phải có:
- `✅ Ultralytics 8.4.127 (bản TSVM_FP32)` và `📁 Dữ liệu: ... {'train': 2455, 'val': 532, 'test': 522}`
- `Tầng 10: C2TSVMamba, 1,811,201 tham số`
- `✅ seed 0: test mask mAP50-95 ...` và `Xong phiên.`
- Dòng `1 epochs completed in X hours`: nhân X với 100 là ước tính giờ của một seed TSVM (dự kiến khoảng 10–10,5 giờ).

**Gửi mình mấy dòng này (hoặc ảnh chụp) trước khi chạy thật.**

## Bước 3. Chạy thật: 8 phiên

Mỗi phiên làm như Bước 2 (Import, Add Input, T4, Internet On), nhưng bấm **Save Version → Save & Run All (Commit)**.

| Notebook | Seed | Ước tính |
|---|---|---|
| `h3-yolo-s0` | YOLO 0 | ~4,4 giờ |
| `h3-yolo-s1-2` | YOLO 1, 2 | ~8,8 giờ |
| `h3-yolo-s3-4` | YOLO 3, 4 | ~8,8 giờ |
| `h3-tsvm-s0` … `h3-tsvm-s4` | TSVM 0 … 4, mỗi phiên 1 seed | ~10–10,5 giờ mỗi phiên |

Tổng khoảng 75 giờ GPU. Mỗi tài khoản có khoảng 30 giờ mỗi tuần, nên một tài khoản chạy được 2 phiên TSVM, hoặc 1 phiên TSVM và 1–2 phiên YOLO. Quota reset thứ Bảy.

Ước tính lấy từ giờ thật của 60 lượt trước, ngoại suy theo số ảnh train (2.455 thay vì 880–1.040).

## Bước 3b. Seed 5–9 (thêm 22/11): 10 phiên

Seed 0–4 đã xong, kết quả nằm trong `../ket_qua/`. Giờ thật trên T4 lâu hơn ước tính: TSVM 11,3–11,5 giờ/seed, YOLO 5,6–6,1 giờ/seed. Nếu ghép 2 seed YOLO vào một phiên thì vượt mốc 11,8 giờ, notebook sẽ tự bỏ qua seed thứ hai. Vì vậy **mỗi phiên chỉ chạy 1 seed**, cho cả hai mô hình.

| Notebook | Seed | Ước tính |
|---|---|---|
| `h3-yolo-s5` … `h3-yolo-s9` | YOLO 5 … 9 | ~6 giờ mỗi phiên |
| `h3-tsvm-s5` … `h3-tsvm-s9` | TSVM 5 … 9 | ~11,3–11,5 giờ mỗi phiên |

- **Tổng:** khoảng 30 giờ YOLO + 57 giờ TSVM = **khoảng 87 giờ GPU**.
- **Chia tài khoản:** mỗi tài khoản có khoảng 30 giờ/tuần, nên chạy được 2 phiên TSVM + 1 phiên YOLO, hoặc 5 phiên YOLO.
- **Cách chạy:** dùng cùng dataset `h3-gop-3tap-tsvm`, không phải tải lại. Các bước giống Bước 3: Import, Add Input, GPU T4 x2, Internet On, Save & Run All.
- **Dừng an toàn:** ở seed 0–4, TSVM seed 1 và 3 dừng an toàn ở epoch 99/100 vì phiên chậm. Seed mới cũng có thể như vậy. Nếu gặp, kết quả vẫn được lưu đủ, xử lý theo mục "Nếu một seed bị dừng an toàn" bên dưới.

## Mốc an toàn 11,8 giờ

- **Trước mỗi seed** (phiên YOLO 2 seed): nếu không kịp xong thì bỏ qua seed đó và in `⏸ Bỏ qua seed ...`.
- **Cuối mỗi epoch:** nếu thời gian phiên + 1,1 × thời gian epoch vừa rồi + 15 phút dự phòng vượt 11,8 giờ, notebook **dừng huấn luyện**, in `⏸ DỪNG AN TOÀN sau epoch N`, chép `last.pt` thành `weights/last_resume.pt` (còn trạng thái optimizer) và ghi `CHUA_XONG.txt`. Sau đó vẫn đánh giá test và nén zip, nên kết quả không mất.
- Mỗi seed xong được nén thành zip riêng ngay, nên phiên 2 seed có bị cắt thì seed đầu vẫn còn.

## Bước 4. Tải kết quả

Ở tab **Output** của mỗi phiên, tải `runs_H3_Gop_3tap_<YOLO26s_seg|TSVM_FP32>_s<N>.zip` về một chỗ, báo mình. Mình giải nén vào `Huong3_GopDuLieu/ket_qua/`.

## Nếu một seed bị dừng an toàn (có `CHUA_XONG.txt`)

1. Tạo một dataset mới (Private) từ zip kết quả của seed đó, ví dụ `h3-tsvm-s2-dung`.
2. Import `h3-chay-tiep.ipynb`, sửa `MODEL` và `SEEDS` ở cell đầu (ví dụ `"tsvm"`, `[2]`).
3. Add Input cả `h3-gop-3tap-tsvm` lẫn dataset vừa tạo, T4, Internet On, Save & Run All.
4. Notebook chạy tiếp từ epoch kế tiếp đến 100 (`resume=True`), rồi đánh giá test và nén zip cùng tên. Mình sẽ ghi seed nào phải chạy tiếp, vì kết quả có thể lệch rất nhẹ so với chạy liền.

## Bước 3c (23/11). Chạy tiếp 4 seed TSVM trong một phiên

TSVM s1 và s3 dừng sau epoch 99, còn s6 và s8 dừng sau epoch 98. Tổng còn 6 epoch, khoảng 1–1,5 giờ trên T4.

1. Tải `1_dataset/h3-chay-tiep-4seed.zip` (683 MB) lên Kaggle, đặt tên dataset `h3-chay-tiep-4seed`, chọn **Private**.
   - Zip gồm 4 thư mục lượt, mỗi thư mục có `weights/last_resume.pt`, `best.pt`, `results.csv` và `args.yaml`.
   - Không kèm `last.pt` cũ vì tệp này đã mất trạng thái optimizer.
2. Import `3_notebooks/h3-chay-tiep-4seed.ipynb`, không cần sửa gì.
3. Add Input cả `h3-gop-3tap-tsvm` lẫn `h3-chay-tiep-4seed`. Chọn GPU T4 x2, bật Internet, rồi Save & Run All.
4. Trước khi chạy, notebook kiểm tra đủ checkpoint cho cả 4 seed và in số epoch còn thiếu của từng seed. Sau đó notebook chạy lần lượt s1 → s3 → s6 → s8. Với mỗi seed, notebook:
   - chạy tiếp đến epoch 100 với `resume=True`;
   - kiểm tra `results.csv` đã tới epoch 100;
   - đánh giá test và ghi `chay_tiep=True` vào `test_metrics.json`;
   - nén zip **cùng tên với zip cũ**.
5. Nếu một seed lỗi, notebook vẫn chạy các seed còn lại. Dòng cuối in danh sách seed đã đủ 100 epoch và seed bị lỗi.
6. Tải 4 zip `runs_H3_Gop_3tap_TSVM_FP32_s{1,3,6,8}.zip` về, rồi báo mình. Mình sẽ thay bản cũ, và bản cũ được chuyển sang `_Luu_Tru_Ngoai`.

Phần nạp checkpoint và dựng trainer đã chạy thử trên máy với s6 (chưa huấn luyện).
- Ultralytics báo `Resuming ... from epoch 99 to 100 total epochs`.
- Mosaic đã tắt, optimizer và best_fitness được nạp lại đúng.

## Lỗi thường gặp

| Thông báo | Cách xử lý |
|---|---|
| `❌ Chưa thêm dataset ...` | Thiếu Input `h3-gop-3tap-tsvm` |
| `❌ Tải chưa đủ` | Bật Settings → Internet → On rồi chạy lại |
| `AssertionError` ở số ảnh | Dataset bị giải nén sai. Báo mình |
| Queued lâu | Chạy tương tác (Run All), để trang mở. Xong thì Save Version → Quick Save |

## Lưu ý

- Không đổi `SEEDS`, không chạy seed ngoài 0–9, không chạy lại seed đã có kết quả.
- Chỉ chọn checkpoint trên val. Tập test chỉ được đánh giá một lần sau cùng (notebook tự làm).
- Không để dataset Public.
- Không đẩy gì lên notebook `okeydokey0805/vm-unet-640-chaythu`.
