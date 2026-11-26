"""Hướng 3: tạo gói Kaggle (zip dataset + notebook) cho YOLO26s-seg và TSVM, seed 0–9.

Ghi ../Goi_Kaggle_H3/ (dừng nếu đã có zip):
  1_dataset/h3-gop-3tap-tsvm.zip   H3_Gop_3tap/ (images, labels, data_h3_3tap.yaml, kiem_tra_nhan.csv; KHÔNG kèm masks/)
                                   + ultralytics_TSVM_FP32/ chép nguyên từ zip hướng 2 (cùng mã đã chạy 20 lượt hướng 2)
  3_notebooks/*.ipynb              chia phiên theo người dùng chốt 19/11:
                                   YOLO: s0 riêng 1 phiên, s1+s2, s3+s4; TSVM: mỗi seed 1 phiên; 1 notebook chạy thử TSVM
                                   1 epoch; 1 notebook chạy tiếp (resume) dùng khi một seed bị dừng ở mốc an toàn.
                                   22/11: thêm seed 5–9 (người dùng chốt đủ 10 seed/mô hình). YOLO thực tế 5,6–6,1 giờ/seed
                                   nên 2 seed không vừa một phiên 11,8 giờ: seed 5–9 mỗi phiên 1 seed cho cả hai mô hình.
Siêu tham số giữ như 40 lượt trong báo cáo và 20 lượt hướng 2. Khác hướng 2:
- Dữ liệu gộp 3 tập (train 2.455 / val 532 / test 522), số ảnh được kiểm bằng assert.
- Mốc an toàn 11,8 giờ (người dùng chốt): ngoài việc không bắt đầu seed mới khi không kịp, còn có callback cuối mỗi
  epoch: nếu thời gian phiên + 1,1 x epoch vừa rồi + 0,25 giờ dự phòng (đánh giá test, nén zip) vượt 11,8 giờ thì dừng
  huấn luyện, chép last.pt (còn optimizer) thành last_resume.pt để phiên sau chạy tiếp, ghi CHUA_XONG.txt.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
H3 = HERE.parent
DATA = H3 / "du_lieu" / "H3_Gop_3tap"
OUT = H3 / "Goi_Kaggle_H3"
ZIP_H2 = H3.parent / "Huong2_Chia3Tap" / "Goi_Kaggle_H2" / "1_dataset" / "kvasir-bg20-3tap-tsvm.zip"
ZIP_RA = OUT / "1_dataset" / "h3-gop-3tap-tsvm.zip"
DEM = {"train": 2455, "val": 532, "test": 522}

PHIEN = [
    ("h3-chay-thu-tsvm", "tsvm", [0]),
    ("h3-yolo-s0", "yolo", [0]),
    ("h3-yolo-s1-2", "yolo", [1, 2]),
    ("h3-yolo-s3-4", "yolo", [3, 4]),
    ("h3-tsvm-s0", "tsvm", [0]),
    ("h3-tsvm-s1", "tsvm", [1]),
    ("h3-tsvm-s2", "tsvm", [2]),
    ("h3-tsvm-s3", "tsvm", [3]),
    ("h3-tsvm-s4", "tsvm", [4]),
    ("h3-yolo-s5", "yolo", [5]),
    ("h3-yolo-s6", "yolo", [6]),
    ("h3-yolo-s7", "yolo", [7]),
    ("h3-yolo-s8", "yolo", [8]),
    ("h3-yolo-s9", "yolo", [9]),
    ("h3-tsvm-s5", "tsvm", [5]),
    ("h3-tsvm-s6", "tsvm", [6]),
    ("h3-tsvm-s7", "tsvm", [7]),
    ("h3-tsvm-s8", "tsvm", [8]),
    ("h3-tsvm-s9", "tsvm", [9]),
]

CELL0 = '''# ============================================================
# CELL 0: CẤU HÌNH PHIÊN (HƯỚNG 3: DỮ LIỆU GỘP CHIA 3 TẬP)
# ============================================================
MODEL = "{model}"          # "yolo" = YOLO26s-seg | "tsvm" = YOLO26s-seg + TSVM
SEEDS = {seeds}            # các seed chạy lần lượt trong phiên này (tổng thể: seed 0–9)
CHAY_THU = {chay_thu}       # True: 1 epoch, 1 seed, để kiểm tra notebook trước khi chạy thật
GIOI_HAN_GIO = 11.8        # mốc an toàn: Kaggle cắt phiên ở 12 giờ
DU_PHONG_GIO = 0.25        # để dành cho đánh giá test + nén zip sau khi dừng huấn luyện

assert MODEL in ("yolo", "tsvm")
assert all(s in range(10) for s in SEEDS), "Hướng 3 chỉ dùng seed 0–9"
EPOCHS = 1 if CHAY_THU else 100
if CHAY_THU:
    SEEDS = SEEDS[:1]
print(f"MODEL={{MODEL}}, SEEDS={{SEEDS}}, EPOCHS={{EPOCHS}}")
'''

CELL1 = '''# ============================================================
# CELL 1: TÌM INPUT, NẠP ULTRALYTICS BẢN TSVM_FP32, TẠO DATA YAML (TRAIN / VAL / TEST)
# ============================================================
import os, sys, glob, time, json, shutil, yaml
T_PHIEN = time.time()

def tim(mau):
    return sorted(glob.glob(mau, recursive=True))

_repo = tim("/kaggle/input/**/ultralytics_TSVM_FP32/PATCH_FP32.md")
_data = tim("/kaggle/input/**/H3_Gop_3tap/images/test")
assert _repo, "❌ Chưa thêm dataset chứa thư mục ultralytics_TSVM_FP32"
assert _data, "❌ Chưa thêm dataset h3-gop-3tap-tsvm (thư mục H3_Gop_3tap)"
CUSTOM_REPO_PATH = os.path.dirname(_repo[0])
DATASET_ROOT = os.path.dirname(os.path.dirname(_data[0]))

if not os.path.exists("/kaggle/working/ultralytics"):
    os.symlink(CUSTOM_REPO_PATH, "/kaggle/working/ultralytics")
for p in ["/kaggle/working", CUSTOM_REPO_PATH]:
    if p not in sys.path:
        sys.path.insert(0, p)

import ultralytics
assert ultralytics.__version__ == "8.4.127", ultralytics.__version__
assert ultralytics.__file__.startswith("/kaggle/working/ultralytics"), ultralytics.__file__
from ultralytics.nn.modules import C2TSVMamba
print(f"✅ Ultralytics {ultralytics.__version__} (bản TSVM_FP32) từ {ultralytics.__file__}")

dem = {s: len(os.listdir(f"{DATASET_ROOT}/images/{s}")) for s in ("train", "val", "test")}
dem_nhan = {s: len(os.listdir(f"{DATASET_ROOT}/labels/{s}")) for s in ("train", "val", "test")}
assert dem == DEM_DUNG and dem_nhan == DEM_DUNG, (dem, dem_nhan)
print("📁 Dữ liệu:", DATASET_ROOT, dem)

DATA_YAML = "/kaggle/working/data_h3_3tap.yaml"
with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump({"path": DATASET_ROOT, "train": "images/train", "val": "images/val", "test": "images/test",
                    "nc": 1, "names": {0: "polyp"}}, f, sort_keys=False, allow_unicode=True)

if MODEL == "tsvm":
    SRC_YAML = f"{CUSTOM_REPO_PATH}/cfg/models/26/yolo26-seg-TopologyShapeVMamba.yaml"
    # Tên tệp PHẢI bắt đầu bằng "yolo26s": Ultralytics đoán scale từ tên tệp
    MODEL_YAML = "/kaggle/working/yolo26s-seg-TopologyShapeVMamba.yaml"
    with open(SRC_YAML, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["scale"] = "s"
    with open(MODEL_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    TEN = "TSVM_FP32"
else:
    MODEL_YAML = None
    TEN = "YOLO26s_seg"
'''.replace("DEM_DUNG", json.dumps(DEM))

CELL2 = '''# ============================================================
# CELL 2: TẢI TRỌNG SỐ yolo26s-seg.pt (CẦN Settings > Internet: ON)
# ============================================================
WEIGHT_PATH = "/kaggle/working/yolo26s-seg.pt"
URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s-seg.pt"
if os.path.exists(WEIGHT_PATH) and os.path.getsize(WEIGHT_PATH) < 20 * 1024 * 1024:
    os.remove(WEIGHT_PATH)
if not os.path.exists(WEIGHT_PATH):
    !curl -L --retry 5 --retry-delay 2 -C - "{URL}" -o "{WEIGHT_PATH}"
size_mb = os.path.getsize(WEIGHT_PATH) / 2**20
print(f"Dung lượng: {size_mb:.2f} MB")
assert size_mb > 20.0, "❌ Tải chưa đủ, chạy lại cell này (Settings > Internet: ON)"
'''

CELL3 = '''# ============================================================
# CELL 3: KIỂM TRA CẤU TRÚC MÔ HÌNH TRƯỚC KHI TRAIN (~10 GIÂY)
# ============================================================
from ultralytics import YOLO
_m = YOLO(MODEL_YAML).load(WEIGHT_PATH) if MODEL == "tsvm" else YOLO(WEIGHT_PATH)
l10 = _m.model.model[10]
n_all = sum(p.numel() for p in _m.model.parameters())
n_l10 = sum(p.numel() for p in l10.parameters())
print(f"Tầng 10: {type(l10).__name__}, {n_l10:,} tham số; toàn mô hình {n_all:,}")
if MODEL == "tsvm":
    assert type(l10).__name__ == "C2TSVMamba" and n_l10 == 1_811_201, "❌ Tầng 10 không đúng TSVM đầy đủ"
else:
    assert type(l10).__name__ == "C2PSA", "❌ Baseline phải giữ C2PSA ở tầng 10"
print("✅ Đúng cấu trúc")
del _m
'''

CELL4 = '''# ============================================================
# CELL 4: HUẤN LUYỆN TỪNG SEED (GIỐNG HỆT BÁO CÁO) + MỐC AN TOÀN 11,8 GIỜ + ĐÁNH GIÁ TRÊN TẬP TEST
# ============================================================
import random, numpy as np, torch
WORKERS, BATCH = 2, 8
thoi_gian_seed, bo_qua, dung_som = [], [], []

def gio_phien():
    return (time.time() - T_PHIEN) / 3600

def tao_callback_dung(run_dir_ref, trang_thai):
    """Cuối mỗi epoch (sau khi đã lưu last.pt): nếu epoch kế tiếp có thể làm phiên vượt mốc thì dừng."""
    def cb(trainer):
        if trang_thai["da_dung"]:
            return
        now = time.time()
        ep = now - trang_thai["t_epoch"]
        trang_thai["t_epoch"] = now
        if gio_phien() + 1.1 * ep / 3600 + DU_PHONG_GIO > GIOI_HAN_GIO and trainer.epoch + 1 < trainer.epochs:
            trang_thai["da_dung"] = True
            trainer.stop = True
            shutil.copy2(trainer.last, trainer.last.parent / "last_resume.pt")   # last.pt sắp bị bỏ optimizer
            with open(f"{trainer.save_dir}/CHUA_XONG.txt", "w") as f:
                f.write(f"Dừng ở mốc an toàn sau epoch {trainer.epoch + 1}/{trainer.epochs}, phiên {gio_phien():.2f} giờ.\\n"
                        "Chạy tiếp bằng notebook h3-chay-tiep với weights/last_resume.pt.\\n")
            print(f"⏸ DỪNG AN TOÀN sau epoch {trainer.epoch + 1}: phiên {gio_phien():.2f} giờ, epoch vừa rồi {ep / 60:.1f} phút")
    return cb

for SEED in SEEDS:
    du_kien = max(thoi_gian_seed) * 1.1 if thoi_gian_seed else 0.0
    if thoi_gian_seed and gio_phien() + du_kien + DU_PHONG_GIO > GIOI_HAN_GIO:
        print(f"⏸ Bỏ qua seed {SEED}: phiên đã {gio_phien():.1f} giờ, seed trước {du_kien / 1.1:.1f} giờ. Chạy seed này ở phiên khác.")
        bo_qua.append(SEED)
        continue
    t0 = time.time()
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)

    model = YOLO(MODEL_YAML).load(WEIGHT_PATH) if MODEL == "tsvm" else YOLO(WEIGHT_PATH)
    RUN_NAME = f"H3_Gop_3tap_{TEN}_s{SEED}" + ("_chaythu" if CHAY_THU else "")
    trang_thai = {"da_dung": False, "t_epoch": time.time()}
    model.add_callback("on_train_start", lambda tr: trang_thai.update(t_epoch=time.time()))
    model.add_callback("on_fit_epoch_end", tao_callback_dung(None, trang_thai))
    model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=640, batch=BATCH, device=0, workers=WORKERS, cache=False,
                amp=False, optimizer="AdamW", seed=SEED, lr0=0.001, warmup_epochs=5.0, deterministic=True,
                patience=100, close_mosaic=10, project="/kaggle/working/runs", name=RUN_NAME, exist_ok=True, plots=True)
    run_dir = f"/kaggle/working/runs/{RUN_NAME}"
    del model
    torch.cuda.empty_cache()
    if trang_thai["da_dung"]:
        dung_som.append(SEED)

    # Đánh giá best.pt trên tập TEST (chỉ một lần, sau khi đã chọn checkpoint trên val). Seed dừng sớm vẫn đánh giá
    # để có số tạm, nhưng số chính thức lấy sau khi chạy tiếp đủ 100 epoch.
    best = YOLO(f"{run_dir}/weights/best.pt")
    m = best.val(data=DATA_YAML, split="test", imgsz=640, batch=BATCH, half=False, device=0, plots=False,
                 project="/kaggle/working/runs", name=RUN_NAME + "_test", exist_ok=True)
    kq = {k: float(v) for k, v in m.results_dict.items()}
    kq.update(seed=SEED, model=TEN, gio_huan_luyen=(time.time() - t0) / 3600, dung_som=trang_thai["da_dung"],
              best_pt_mb=os.path.getsize(f"{run_dir}/weights/best.pt") / 2**20,
              gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    with open(f"{run_dir}/test_metrics.json", "w") as f:
        json.dump(kq, f, indent=1)
    print(f"✅ seed {SEED}: test mask mAP50-95 {kq.get('metrics/mAP50-95(M)', float('nan')):.4f}, "
          f"{kq['gio_huan_luyen']:.2f} giờ" + ("  (DỪNG SỚM, cần chạy tiếp)" if trang_thai["da_dung"] else ""))
    del best
    torch.cuda.empty_cache()

    shutil.make_archive(f"/kaggle/working/runs_{RUN_NAME}", "zip", "/kaggle/working/runs", RUN_NAME)
    thoi_gian_seed.append((time.time() - t0) / 3600)

print(f"\\nXong phiên. Đã chạy: {[s for s in SEEDS if s not in bo_qua]}; bỏ qua: {bo_qua}; dừng sớm: {dung_som}; "
      f"tổng {gio_phien():.2f} giờ")
'''

CELL_TIEP0 = '''# ============================================================
# CHẠY TIẾP MỘT SEED BỊ DỪNG Ở MỐC AN TOÀN (CHỈ DÙNG KHI CÓ CHUA_XONG.txt)
# Thêm Input: dataset h3-gop-3tap-tsvm VÀ dataset chứa last_resume.pt của seed đó (tạo từ zip kết quả phiên trước).
# ============================================================
MODEL = "tsvm"             # sửa theo seed bị dừng
SEEDS = [0]                # đúng 1 seed bị dừng
CHAY_THU = False
GIOI_HAN_GIO = 11.8
DU_PHONG_GIO = 0.25
assert len(SEEDS) == 1
'''

CELL_TIEP4 = '''# ============================================================
# CELL 4 (CHẠY TIẾP): resume=True từ last_resume.pt, giữ nguyên tên lượt, rồi đánh giá test
# ============================================================
import torch
SEED = SEEDS[0]
RUN_NAME = f"H3_Gop_3tap_{TEN}_s{SEED}"
_ck = [p for p in tim("/kaggle/input/**/last_resume.pt") if f"_s{SEED}" in p or len(tim("/kaggle/input/**/last_resume.pt")) == 1]
assert len(_ck) == 1, f"❌ Cần đúng 1 last_resume.pt của seed {SEED}: {_ck}"
run_dir = f"/kaggle/working/runs/{RUN_NAME}"
os.makedirs(f"{run_dir}/weights", exist_ok=True)
# chép cả thư mục lượt cũ (results.csv, best.pt, args.yaml...) để Ultralytics ghi tiếp vào đúng chỗ
_src = os.path.dirname(os.path.dirname(_ck[0]))
shutil.copytree(_src, run_dir, dirs_exist_ok=True)
shutil.copy2(_ck[0], f"{run_dir}/weights/last.pt")
ck = torch.load(f"{run_dir}/weights/last.pt", map_location="cpu", weights_only=False)
ck["train_args"].update(data=DATA_YAML, project="/kaggle/working/runs", name=RUN_NAME, save_dir=run_dir)
torch.save(ck, f"{run_dir}/weights/last.pt")
print("Chạy tiếp từ epoch", ck["epoch"] + 2, "/ 100")
t0 = time.time()
YOLO(f"{run_dir}/weights/last.pt").train(resume=True)
best = YOLO(f"{run_dir}/weights/best.pt")
m = best.val(data=DATA_YAML, split="test", imgsz=640, batch=8, half=False, device=0, plots=False,
             project="/kaggle/working/runs", name=RUN_NAME + "_test", exist_ok=True)
kq = {k: float(v) for k, v in m.results_dict.items()}
kq.update(seed=SEED, model=TEN, gio_phien_chay_tiep=(time.time() - t0) / 3600, chay_tiep=True,
          gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
json.dump(kq, open(f"{run_dir}/test_metrics.json", "w"), indent=1)
if os.path.exists(f"{run_dir}/CHUA_XONG.txt"):
    os.rename(f"{run_dir}/CHUA_XONG.txt", f"{run_dir}/DA_CHAY_TIEP.txt")
shutil.make_archive(f"/kaggle/working/runs_{RUN_NAME}", "zip", "/kaggle/working/runs", RUN_NAME)
print(f"✅ seed {SEED} chạy tiếp xong: test mask mAP50-95 {kq.get('metrics/mAP50-95(M)', float('nan')):.4f}")
'''

# 23/11: chạy tiếp nhiều seed TSVM bị dừng ở mốc an toàn trong MỘT phiên (s1, s3 dừng sau epoch 99; s6, s8 sau 98)
SEEDS_TIEP = [1, 3, 6, 8]
TIEP_DATASET = "h3-chay-tiep-4seed"

CELL_TIEPN0 = '''# ============================================================
# CHẠY TIẾP 4 SEED TSVM BỊ DỪNG Ở MỐC AN TOÀN, LẦN LƯỢT TRONG MỘT PHIÊN
# Input: dataset h3-gop-3tap-tsvm (dữ liệu + mã) VÀ dataset TIEP_DATASET (4 thư mục lượt có weights/last_resume.pt)
# Còn thiếu: s1, s3 1 epoch; s6, s8 2 epoch -> tổng 6 epoch, khoảng 1-1,5 giờ trên T4
# ============================================================
MODEL = "tsvm"
SEEDS = SEEDS_TIEP
CHAY_THU = False
GIOI_HAN_GIO = 11.8
DU_PHONG_GIO = 0.25
assert all(s in range(10) for s in SEEDS) and len(set(SEEDS)) == len(SEEDS)
print("Chạy tiếp các seed:", SEEDS)
'''.replace("SEEDS_TIEP", json.dumps(SEEDS_TIEP)).replace("TIEP_DATASET", TIEP_DATASET)

CELL_TIEPN4 = '''# ============================================================
# CELL 4 (CHẠY TIẾP NHIỀU SEED): mỗi seed resume=True từ last_resume.pt, giữ nguyên tên lượt, đánh giá test, nén zip
# ============================================================
import gc, torch

def tim_ck(seed):
    # khớp đúng tên thư mục lượt (".../H3_Gop_3tap_TSVM_FP32_s1/weights/..."), không lẫn s1 với s10
    return [p for p in tim("/kaggle/input/**/last_resume.pt")
            if p.replace("\\\\", "/").endswith(f"/H3_Gop_3tap_{TEN}_s{seed}/weights/last_resume.pt")]

# Kiểm tra đủ checkpoint cho mọi seed TRƯỚC khi chạy, tránh chạy được nửa phiên mới phát hiện thiếu
CK = {}
for s in SEEDS:
    _c = tim_ck(s)
    assert len(_c) == 1, f"❌ Cần đúng 1 last_resume.pt của seed {s}: {_c}"
    _e = torch.load(_c[0], map_location="cpu", weights_only=False)["epoch"]
    print(f"seed {s}: đã xong {_e + 1}/100 epoch, còn {99 - _e} epoch  <- {_c[0]}")
    assert 0 <= _e < 99, f"❌ seed {s} đã đủ 100 epoch, không cần chạy tiếp"
    CK[s] = _c[0]

xong, loi = [], {}
for SEED in SEEDS:
    t0 = time.time()
    RUN_NAME = f"H3_Gop_3tap_{TEN}_s{SEED}"
    run_dir = f"/kaggle/working/runs/{RUN_NAME}"
    try:
        # chép cả thư mục lượt cũ (results.csv, best.pt, args.yaml...) để Ultralytics ghi tiếp vào đúng chỗ
        shutil.copytree(os.path.dirname(os.path.dirname(CK[SEED])), run_dir, dirs_exist_ok=True)
        shutil.copy2(CK[SEED], f"{run_dir}/weights/last.pt")
        ck = torch.load(f"{run_dir}/weights/last.pt", map_location="cpu", weights_only=False)
        assert ck["train_args"]["seed"] == SEED and ck["train_args"]["epochs"] == 100, ck["train_args"]
        ck["train_args"].update(data=DATA_YAML, project="/kaggle/working/runs", name=RUN_NAME, save_dir=run_dir)
        e0 = ck["epoch"]
        torch.save(ck, f"{run_dir}/weights/last.pt")
        del ck
        print(f"\\n▶ seed {SEED}: chạy tiếp từ epoch {e0 + 2} / 100")
        YOLO(f"{run_dir}/weights/last.pt").train(resume=True)
        gc.collect(); torch.cuda.empty_cache()

        import pandas as pd
        _r = pd.read_csv(f"{run_dir}/results.csv")
        _r.columns = [c.strip() for c in _r.columns]
        assert int(_r["epoch"].iloc[-1]) == 100, f"results.csv chỉ tới epoch {_r['epoch'].iloc[-1]}"
        os.remove(f"{run_dir}/weights/last_resume.pt")   # bản cũ, không còn cần

        best = YOLO(f"{run_dir}/weights/best.pt")
        m = best.val(data=DATA_YAML, split="test", imgsz=640, batch=8, half=False, device=0, plots=False,
                     project="/kaggle/working/runs", name=RUN_NAME + "_test", exist_ok=True)
        kq = {k: float(v) for k, v in m.results_dict.items()}
        kq.update(seed=SEED, model=TEN, chay_tiep=True, chay_tiep_tu_epoch=e0 + 2,
                  gio_phien_chay_tiep=(time.time() - t0) / 3600,
                  best_pt_mb=os.path.getsize(f"{run_dir}/weights/best.pt") / 2**20,
                  gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
        with open(f"{run_dir}/test_metrics.json", "w") as f:
            json.dump(kq, f, indent=1)
        del best, m
        gc.collect(); torch.cuda.empty_cache()
        if os.path.exists(f"{run_dir}/CHUA_XONG.txt"):
            os.rename(f"{run_dir}/CHUA_XONG.txt", f"{run_dir}/DA_CHAY_TIEP.txt")
        shutil.make_archive(f"/kaggle/working/runs_{RUN_NAME}", "zip", "/kaggle/working/runs", RUN_NAME)
        xong.append(SEED)
        print(f"✅ seed {SEED} xong 100 epoch: test mask mAP50-95 {kq.get('metrics/mAP50-95(M)', float('nan')):.4f}, "
              f"{kq['gio_phien_chay_tiep'] * 60:.0f} phút")
    except Exception as e:   # một seed lỗi không làm mất các seed còn lại
        loi[SEED] = repr(e)
        print(f"❌ seed {SEED} lỗi: {e!r}")
        gc.collect(); torch.cuda.empty_cache()

print(f"\\nXong phiên {(time.time() - T_PHIEN) / 3600:.2f} giờ. Đủ 100 epoch: {xong}; lỗi: {loi}")
print("Tải về:", sorted(p for p in os.listdir("/kaggle/working") if p.startswith("runs_") and p.endswith(".zip")))
'''


def tao_dataset_tiep():
    """Gom 4 thư mục lượt (bỏ last.pt đã mất optimizer) thành zip để tải lên Kaggle làm dataset TIEP_DATASET."""
    ra = OUT / "1_dataset" / f"{TIEP_DATASET}.zip"
    if ra.exists():
        print("Zip chạy tiếp đã có, giữ nguyên:", ra)
        return
    n = 0
    with zipfile.ZipFile(ra, "w", zipfile.ZIP_STORED) as zo:
        for s in SEEDS_TIEP:
            ten = f"H3_Gop_3tap_TSVM_FP32_s{s}"
            thu_muc, zip_goc = H3 / "ket_qua" / ten, H3 / f"runs_{ten}.zip"
            if thu_muc.is_dir():
                nguon = [(p.relative_to(thu_muc).as_posix(), p.read_bytes) for p in sorted(thu_muc.rglob("*")) if p.is_file()]
            else:
                zi = zipfile.ZipFile(zip_goc)
                nguon = [(i.filename[len(ten) + 1:], (lambda i=i: zi.read(i))) for i in zi.infolist()
                         if not i.is_dir() and i.filename.startswith(ten + "/")]
            rel = {r for r, _ in nguon}
            assert {"weights/last_resume.pt", "weights/best.pt", "results.csv", "args.yaml"} <= rel, (s, sorted(rel))
            for r, doc in nguon:
                if r == "weights/last.pt":
                    continue
                zo.writestr(f"{ten}/{r}", doc())
                n += 1
    print(f"Zip chạy tiếp: {ra} | {n} tệp | {ra.stat().st_size / 2**20:.1f} MB")


def nb(cells):
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                       "source": c.splitlines(keepends=True)} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def tao_zip():
    if ZIP_RA.exists():
        print("Zip đã có, giữ nguyên:", ZIP_RA)
        return
    ZIP_RA.parent.mkdir(parents=True, exist_ok=True)
    n_code = n_data = 0
    with zipfile.ZipFile(ZIP_H2) as zi, zipfile.ZipFile(ZIP_RA, "w", zipfile.ZIP_STORED) as zo:
        for info in zi.infolist():
            if "ultralytics_TSVM_FP32/" in info.filename and not info.is_dir():
                ten = info.filename[info.filename.index("ultralytics_TSVM_FP32/"):]
                zo.writestr(ten, zi.read(info))
                n_code += 1
        for p in sorted(DATA.rglob("*")):
            rel = p.relative_to(DATA)
            if p.is_dir() or rel.parts[0] == "masks" or rel.parts[0].lower().startswith("new folder"):
                continue
            zo.write(p, f"H3_Gop_3tap/{rel.as_posix()}")
            n_data += 1
    print(f"Zip: {ZIP_RA} | mã {n_code} tệp | dữ liệu {n_data} tệp | {ZIP_RA.stat().st_size / 2**20:.1f} MB")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    tao_zip()
    nbdir = OUT / "3_notebooks"
    nbdir.mkdir(parents=True, exist_ok=True)
    for ten, model, seeds in PHIEN:
        c0 = CELL0.format(model=model, seeds=seeds, chay_thu=str(ten.startswith("h3-chay-thu")))
        (nbdir / f"{ten}.ipynb").write_text(json.dumps(nb([c0, CELL1, CELL2, CELL3, CELL4]), ensure_ascii=False, indent=1),
                                            encoding="utf-8")
    (nbdir / "h3-chay-tiep.ipynb").write_text(json.dumps(nb([CELL_TIEP0, CELL1, CELL2, CELL3, CELL_TIEP4]),
                                                         ensure_ascii=False, indent=1), encoding="utf-8")
    (nbdir / "h3-chay-tiep-4seed.ipynb").write_text(json.dumps(nb([CELL_TIEPN0, CELL1, CELL2, CELL3, CELL_TIEPN4]),
                                                               ensure_ascii=False, indent=1), encoding="utf-8")
    tao_dataset_tiep()
    print("Notebook:", sorted(p.name for p in nbdir.glob("*.ipynb")))


if __name__ == "__main__":
    main()
