"""Sinh notebook Kaggle cho Hướng 2 (BG20 chia 3 tập): YOLO26s-seg và YOLO26s-seg + TSVM, seed 0–9.

Giữ nguyên siêu tham số của 40 lượt trong báo cáo (args.yaml): 640, 100 epoch, batch 8, AdamW lr0 1e-3,
warmup 5, close_mosaic 10, amp=False, deterministic, patience 100, workers 2.
Khác báo cáo:
- Dữ liệu: train 760+120 nền, val 120+40 (chọn checkpoint), test 120+40 (= val cũ, chỉ để báo cáo).
- Cả hai mô hình nạp cùng mã Ultralytics 8.4.127 bản TSVM_FP32 (chép từ Test_Mau/02_ma_nguon).
  Với baseline, mã huấn luyện giống pip 8.4.127; chỉ khác best.pt lưu FP32 thay vì FP16.
- Mỗi phiên chạy nhiều seed liên tiếp; trước mỗi seed kiểm tra còn đủ giờ (GIOI_HAN_GIO), nếu không thì bỏ qua
  seed đó và báo lại để chạy ở phiên sau.
- Sau mỗi seed: validate best.pt trên tập TEST (half=False) và lưu chỉ số vào test_metrics.json.

    C:\\venv_kl_gpu\\Scripts\\python.exe tao_notebook_h2.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "Goi_Kaggle_H2" / "3_notebooks"

# (tên file, MODEL, SEEDS). Ước tính: YOLO ~2,3 giờ/seed, TSVM ~4,6 giờ/seed (log báo cáo, train 880+160 ảnh).
PHIEN = [
    ("h2-yolo-s0-3", "yolo", [0, 1, 2, 3]),
    ("h2-yolo-s4-7", "yolo", [4, 5, 6, 7]),
    ("h2-yolo-s8-9", "yolo", [8, 9]),
    ("h2-tsvm-s0-1", "tsvm", [0, 1]),
    ("h2-tsvm-s2-3", "tsvm", [2, 3]),
    ("h2-tsvm-s4-5", "tsvm", [4, 5]),
    ("h2-tsvm-s6-7", "tsvm", [6, 7]),
    ("h2-tsvm-s8-9", "tsvm", [8, 9]),
    ("h2-chay-thu-tsvm", "tsvm", [0]),
    ("h2-chay-thu-yolo", "yolo", [0]),
]

CELL0 = '''# ============================================================
# CELL 0: CẤU HÌNH PHIÊN (HƯỚNG 2: BG20 CHIA 3 TẬP)
# ============================================================
MODEL = "{model}"          # "yolo" = YOLO26s-seg | "tsvm" = YOLO26s-seg + TSVM
SEEDS = {seeds}            # các seed chạy lần lượt trong phiên này (tổng thể: seed 0–9)
CHAY_THU = {chay_thu}       # True: 1 epoch, 1 seed, để kiểm tra notebook trước khi chạy thật
GIOI_HAN_GIO = 11.0        # Kaggle cắt phiên ở 12 giờ: không bắt đầu seed mới nếu không kịp xong trước mốc này

assert MODEL in ("yolo", "tsvm")
assert all(s in range(10) for s in SEEDS), "Hướng 2 chỉ dùng seed 0–9"
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
_data = tim("/kaggle/input/**/Kvasir_BG20_3tap/images/test")
assert _repo, "❌ Chưa thêm dataset chứa thư mục ultralytics_TSVM_FP32"
assert _data, "❌ Chưa thêm dataset chứa Kvasir_BG20_3tap"
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
assert dem == {"train": 880, "val": 160, "test": 160}, dem
print("📁 Dữ liệu:", DATASET_ROOT, dem)

DATA_YAML = "/kaggle/working/data_bg20_3tap.yaml"
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
'''

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
# CELL 4: HUẤN LUYỆN TỪNG SEED (GIỐNG HỆT 40 LƯỢT TRONG BÁO CÁO) + ĐÁNH GIÁ TRÊN TẬP TEST
# ============================================================
import random, numpy as np, torch
WORKERS, BATCH = 2, 8
thoi_gian_seed, bo_qua = [], []

for SEED in SEEDS:
    da_chay = (time.time() - T_PHIEN) / 3600
    du_kien = max(thoi_gian_seed) * 1.1 if thoi_gian_seed else 0.0
    if da_chay + du_kien > GIOI_HAN_GIO:
        print(f"⏸ Bỏ qua seed {SEED}: đã chạy {da_chay:.1f} giờ, seed trước {du_kien / 1.1:.1f} giờ. Chạy seed này ở phiên khác.")
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
    RUN_NAME = f"H2_BG20_3tap_{TEN}_s{SEED}" + ("_chaythu" if CHAY_THU else "")
    model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=640, batch=BATCH, device=0, workers=WORKERS, cache=False,
                amp=False, optimizer="AdamW", seed=SEED, lr0=0.001, warmup_epochs=5.0, deterministic=True,
                patience=100, close_mosaic=10, project="/kaggle/working/runs", name=RUN_NAME, exist_ok=True, plots=True)
    run_dir = f"/kaggle/working/runs/{RUN_NAME}"
    del model
    torch.cuda.empty_cache()

    # Đánh giá best.pt trên tập TEST (chỉ một lần, sau khi đã chọn checkpoint trên val)
    best = YOLO(f"{run_dir}/weights/best.pt")
    m = best.val(data=DATA_YAML, split="test", imgsz=640, batch=BATCH, half=False, device=0, plots=False,
                 project="/kaggle/working/runs", name=RUN_NAME + "_test", exist_ok=True)
    kq = {k: float(v) for k, v in m.results_dict.items()}
    kq.update(seed=SEED, model=TEN, gio_huan_luyen=(time.time() - t0) / 3600,
              best_pt_mb=os.path.getsize(f"{run_dir}/weights/best.pt") / 2**20,
              gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
    with open(f"{run_dir}/test_metrics.json", "w") as f:
        json.dump(kq, f, indent=1)
    print(f"✅ seed {SEED}: test mask mAP50-95 {kq.get('metrics/mAP50-95(M)', float('nan')):.4f}, "
          f"{kq['gio_huan_luyen']:.2f} giờ")
    del best
    torch.cuda.empty_cache()

    shutil.make_archive(f"/kaggle/working/runs_{RUN_NAME}", "zip", "/kaggle/working/runs", RUN_NAME)
    thoi_gian_seed.append((time.time() - t0) / 3600)

print(f"\\nXong phiên. Đã chạy: {[s for s in SEEDS if s not in bo_qua]}; bỏ qua: {bo_qua}; "
      f"tổng {(time.time() - T_PHIEN) / 3600:.2f} giờ")
'''


def nb(cells):
    return {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                       "source": c.splitlines(keepends=True)} for c in cells],
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "nbformat": 4, "nbformat_minor": 4}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for ten, model, seeds in PHIEN:
        c0 = CELL0.format(model=model, seeds=seeds, chay_thu=str(ten.startswith("h2-chay-thu")))
        (OUT / f"{ten}.ipynb").write_text(json.dumps(nb([c0, CELL1, CELL2, CELL3, CELL4]), ensure_ascii=False, indent=1),
                                          encoding="utf-8")
        print(OUT / f"{ten}.ipynb")


if __name__ == "__main__":
    main()
