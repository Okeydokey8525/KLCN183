"""Hướng 4: tạo gói Kaggle cho mốc 2 (kiểm tra kernel selective_scan thật + đo tốc độ trên T4).

Ghi ../Goi_Kaggle_H4/ (không ghi đè zip đã có):
  1_dataset/h4-tsvm-cuda.zip
      ultralytics_TSVM_CUDA/        mã Hướng 4 (SS2D hai nhánh, chọn bằng TSVM_SCAN)
      Kvasir_YOLO_SEG_BG20/         dữ liệu báo cáo: train 1.040 / val 160 (dùng lại cho mốc 3)
      trong_so_bao_cao/             best.pt seed 0 của báo cáo: baseline_s0_best.pt, tsvm_fp32_s0_best.pt
      ma_kiem_tra/h4_kiem_tra.py    các phép kiểm tra và đo (A–F)
  3_notebooks/h4-moc2-kiem-tra-kernel.ipynb
Test_Mau chỉ đọc.
"""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
H4 = HERE.parent
DL = H4.parents[1]
TM = DL / "Test_Mau"
CODE = H4 / "ma_nguon" / "ultralytics_TSVM_CUDA"
DATA = TM / "01_du_lieu" / "Kvasir_YOLO_SEG_BG20"
KT = H4 / "Goi_Kaggle_H4" / "ma_kiem_tra" / "h4_kiem_tra.py"
W_BASE = TM / "03_ket_qua_huan_luyen" / "Baseline_YOLO26s_seg" / "Kvasir_BG20_Baseline_YOLO26s_seg_s0_w2" / "weights" / "best.pt"
W_TSVM = TM / "03_ket_qua_huan_luyen" / "TSVM_FP32" / "Kvasir_BG20_YOLO26s_seg_TSVM_FP32_s0_w2" / "weights" / "best.pt"
OUT = H4 / "Goi_Kaggle_H4"
ZIP_RA = OUT / "1_dataset" / "h4-tsvm-cuda.zip"
DEM = {"train": 1040, "val": 160}

CELL0 = r'''# ============================================================
# HƯỚNG 4 - MỐC 2: KIỂM TRA KERNEL selective_scan (mamba_ssm) VÀ ĐO TỐC ĐỘ TRÊN T4
# Settings: Accelerator = GPU T4 x2, Internet = On. Input: dataset h4-tsvm-cuda.
# ============================================================
import os, sys, glob, time, json, shutil, subprocess
import torch
T_PHIEN = time.time()
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| Python", sys.version.split()[0])
print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
subprocess.run("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv", shell=True)
subprocess.run("nvcc --version | tail -n 2", shell=True)
'''

CELL1 = r'''# ============================================================
# CELL 1: CÀI mamba_ssm (KERNEL selective_scan). Gói tự tìm wheel dựng sẵn khớp torch/CUDA; không có thì tự biên dịch
# (có thể 20-60 phút). Wheel được giữ lại trong /kaggle/working/wheels để mốc 3 dùng lại, khỏi biên dịch lần nữa.
# ============================================================
W = "/kaggle/working/wheels"
os.makedirs(W, exist_ok=True)
LOG_CAI = "/kaggle/working/cai_mamba_ssm_log.txt"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops", "ninja", "packaging"], check=False)

def da_co_kernel():
    # Phải import torch TRƯỚC: selective_scan_cuda cần libc10.so do torch nạp (lần chạy 24/11 báo nhầm vì thiếu dòng này)
    r = subprocess.run([sys.executable, "-c", "import torch, selective_scan_cuda, mamba_ssm; "
                        "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print(mamba_ssm.__version__)"],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-600:]

ok, msg = da_co_kernel()
thu = []
for goi in (["mamba-ssm", "mamba-ssm==2.2.4", "mamba-ssm==2.2.2"] if not ok else []):
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "wheel", goi, "--no-build-isolation", "--no-deps", "-w", W, "-v"],
                       capture_output=True, text=True)
    with open(LOG_CAI, "a") as f:
        f.write(f"\n===== pip wheel {goi} (mã thoát {r.returncode}, {(time.time() - t0) / 60:.1f} phút)\n{r.stdout}\n{r.stderr}\n")
    whl = sorted(glob.glob(f"{W}/mamba_ssm-*.whl"))
    dung_san = [l for l in (r.stdout + r.stderr).splitlines() if "wheel" in l.lower() and ("http" in l or "Raw wheel" in l)]
    print(f"{goi}: mã thoát {r.returncode}, {(time.time() - t0) / 60:.1f} phút, wheel {[os.path.basename(w) for w in whl]}")
    print("   ", "\n    ".join(dung_san[-3:]))
    if r.returncode == 0 and whl:
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", whl[-1]], check=True)
        ok, msg = da_co_kernel()
        thu.append((goi, ok))
        if ok:
            break
    else:
        print("\n".join((r.stdout + r.stderr).splitlines()[-25:]))
        thu.append((goi, False))
print("Đã thử:", thu)
print(msg)
assert ok, "❌ Không cài được kernel mamba_ssm. Tải /kaggle/working/cai_mamba_ssm_log.txt gửi lại để xem lỗi."
print(f"✅ Có kernel selective_scan_cuda, mamba_ssm {msg.splitlines()[-1]}")
'''

CELL2 = r'''# ============================================================
# CELL 2: NẠP ULTRALYTICS BẢN HƯỚNG 4 (ultralytics_TSVM_CUDA), KIỂM TRA KERNEL ĐƯỢC DÙNG, TẠO DATA YAML
# ============================================================
import yaml
def tim(mau):
    return sorted(glob.glob(mau, recursive=True))

_repo = tim("/kaggle/input/**/ultralytics_TSVM_CUDA/PATCH_FP32.md")
_data = tim("/kaggle/input/**/Kvasir_YOLO_SEG_BG20/images/val")
_kt = tim("/kaggle/input/**/ma_kiem_tra/h4_kiem_tra.py")
_w = tim("/kaggle/input/**/trong_so_bao_cao/tsvm_fp32_s0_best.pt")
assert _repo and _data and _kt and _w, "❌ Chưa thêm dataset h4-tsvm-cuda (Add Input)"
CUSTOM_REPO_PATH = os.path.dirname(_repo[0])
DATASET_ROOT = os.path.dirname(os.path.dirname(_data[0]))
W_TSVM = _w[0]
W_BASE = os.path.join(os.path.dirname(W_TSVM), "baseline_s0_best.pt")

if not os.path.exists("/kaggle/working/ultralytics"):
    os.symlink(CUSTOM_REPO_PATH, "/kaggle/working/ultralytics")
for p in ["/kaggle/working", os.path.dirname(_kt[0])]:
    if p not in sys.path:
        sys.path.insert(0, p)
import ultralytics
import ultralytics.nn.modules.topology_shape_vmamba as tsv
assert ultralytics.__version__ == "8.4.127", ultralytics.__version__
assert ultralytics.__file__.startswith("/kaggle/working/ultralytics"), ultralytics.__file__
assert tsv.cuda_kernel_available(), f"❌ Mã Hướng 4 không nạp được kernel: {tsv._SELECTIVE_SCAN_IMPORT_ERROR}"
print(f"✅ Ultralytics {ultralytics.__version__} (bản Hướng 4) từ {ultralytics.__file__}; kernel: có")

dem = {s: len(os.listdir(f"{DATASET_ROOT}/images/{s}")) for s in ("train", "val")}
assert dem == DEM_DUNG, dem
DATA_YAML = "/kaggle/working/data_bg20.yaml"
with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump({"path": DATASET_ROOT, "train": "images/train", "val": "images/val", "nc": 1, "names": {0: "polyp"}}, f)
print("📁 Dữ liệu:", DATASET_ROOT, dem)
import h4_kiem_tra as hk
'''.replace("DEM_DUNG", json.dumps(DEM))

CELL3 = r'''# ============================================================
# CELL 3: CHẠY CÁC PHÉP KIỂM TRA VÀ ĐO (khoảng 15-25 phút)
#   A kernel thật so với PyTorch | B val 160 ảnh | C độ trễ forward | D khối SS2D | E bước huấn luyện + bộ nhớ | F video 160 khung
# ============================================================
OUT_DIR = "/kaggle/working/h4_moc2"
KQ = hk.chay_tat_ca(W_BASE, W_TSVM, f"{DATASET_ROOT}/images/val", DATA_YAML, OUT_DIR, nhanh=False)
'''

CELL4 = r'''# ============================================================
# CELL 4: NÉN KẾT QUẢ ĐỂ TẢI VỀ (tab Output): h4_moc2_ket_qua.zip và wheel mamba_ssm (dùng lại cho mốc 3)
# ============================================================
shutil.rmtree(f"{OUT_DIR}/val_runs", ignore_errors=True)
for f in [LOG_CAI]:
    if os.path.exists(f):
        shutil.copy2(f, OUT_DIR)
shutil.make_archive("/kaggle/working/h4_moc2_ket_qua", "zip", OUT_DIR)
print("Tải về:", sorted(p for p in os.listdir("/kaggle/working") if p.endswith(".zip")), "và", os.listdir(W))
print(f"Xong phiên: {(time.time() - T_PHIEN) / 60:.1f} phút")
'''


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
    for s, n in DEM.items():
        assert len(list((DATA / "images" / s).iterdir())) == n and len(list((DATA / "labels" / s).iterdir())) == n, s
    n_code = n_data = 0
    with zipfile.ZipFile(ZIP_RA, "w", zipfile.ZIP_STORED) as zo:
        for p in sorted(CODE.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts:
                zo.write(p, f"ultralytics_TSVM_CUDA/{p.relative_to(CODE).as_posix()}")
                n_code += 1
        for p in sorted(DATA.rglob("*")):
            if p.is_file() and p.suffix != ".cache":
                zo.write(p, f"Kvasir_YOLO_SEG_BG20/{p.relative_to(DATA).as_posix()}")
                n_data += 1
        zo.write(W_BASE, "trong_so_bao_cao/baseline_s0_best.pt")
        zo.write(W_TSVM, "trong_so_bao_cao/tsvm_fp32_s0_best.pt")
        zo.write(KT, "ma_kiem_tra/h4_kiem_tra.py")
    print(f"Zip: {ZIP_RA} | mã {n_code} tệp | dữ liệu {n_data} tệp | {ZIP_RA.stat().st_size / 2**20:.1f} MB")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    tao_zip()
    nbdir = OUT / "3_notebooks"
    nbdir.mkdir(parents=True, exist_ok=True)
    p = nbdir / "h4-moc2-kiem-tra-kernel.ipynb"
    p.write_text(json.dumps(nb([CELL0, CELL1, CELL2, CELL3, CELL4]), ensure_ascii=False, indent=1), encoding="utf-8")
    print("Notebook:", p)


if __name__ == "__main__":
    main()
