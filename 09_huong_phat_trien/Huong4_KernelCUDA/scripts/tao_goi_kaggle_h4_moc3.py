"""Hướng 4, mốc 3: tạo notebook huấn luyện TSVM seed 0 và 15 bằng kernel selective_scan (mamba_ssm), 100 epoch.

Siêu tham số lấy NGUYÊN từ args.yaml của lượt báo cáo (TSVM_FP32 s0), chỉ đổi đường dẫn/tên lượt.
Input Kaggle: dataset h4-tsvm-cuda (mốc 2, giữ nguyên) + dataset h4-mamba-wheel (wheel 2.3.2.post1 cp313).
Ghi ../Goi_Kaggle_H4/3_notebooks/h4-moc3-huan-luyen-s0.ipynb
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from tao_goi_kaggle_h4 import nb, DEM, TM, OUT  # noqa: E402

ARGS_BC = TM / "03_ket_qua_huan_luyen" / "TSVM_FP32" / "Kvasir_BG20_YOLO26s_seg_TSVM_FP32_s0_w2" / "args.yaml"
BO_KHOA = {"task", "mode", "model", "data", "project", "name", "save_dir", "device", "pretrained", "exist_ok"}
args = {k: v for k, v in yaml.safe_load(ARGS_BC.read_text(encoding="utf-8")).items() if k not in BO_KHOA}
assert args["seed"] == 0 and args["epochs"] == 100 and args["batch"] == 8 and args["amp"] is False

CELL0 = r'''# ============================================================
# HƯỚNG 4 - MỐC 3: HUẤN LUYỆN TSVM SEED 0 BẰNG KERNEL selective_scan (mamba_ssm), 100 EPOCH
# Settings: GPU T4 x2, Internet On (tải yolo26s-seg.pt). Input: h4-tsvm-cuda + h4-mamba-wheel.
# CELL 0: CÀI mamba_ssm TỪ WHEEL ĐÃ DỰNG Ở MỐC 2 (vài giây, không biên dịch lại)
# ============================================================
import os, sys, glob, time, json, shutil, subprocess
import torch
T_PHIEN = time.time()
print("torch", torch.__version__, "| CUDA", torch.version.cuda, "| Python", sys.version.split()[0])
print("GPU:", [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
if not torch.__version__.startswith("2.11.0"):
    print("⚠️ Kaggle đã đổi torch (wheel dựng với 2.11.0+cu130). Nếu ô kiểm tra dưới báo lỗi undefined symbol, gửi lại mình.")

def da_co_kernel():
    r = subprocess.run([sys.executable, "-c", "import torch, selective_scan_cuda, mamba_ssm; "
                        "from mamba_ssm.ops.selective_scan_interface import selective_scan_fn; print(mamba_ssm.__version__)"],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-1500:]

subprocess.run([sys.executable, "-m", "pip", "install", "-q", "einops"], check=False)
ok, msg = da_co_kernel()
if not ok:
    whl = sorted(glob.glob("/kaggle/input/**/mamba_ssm-*.whl", recursive=True))
    assert whl, "❌ Chưa thêm dataset h4-mamba-wheel (Add Input)"
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--force-reinstall", whl[0]], check=True)
    ok, msg = da_co_kernel()
print(msg)
assert ok, "❌ Wheel không nạp được kernel, gửi mình đoạn lỗi in phía trên"
print("✅ Có kernel selective_scan_cuda, mamba_ssm", msg.splitlines()[-1])
'''

CELL1 = r'''# ============================================================
# CELL 1: NẠP ULTRALYTICS BẢN HƯỚNG 4, YAML SCALE s, DATA YAML (giống notebook báo cáo)
# ============================================================
import yaml
def tim(mau):
    return sorted(glob.glob(mau, recursive=True))
_repo = tim("/kaggle/input/**/ultralytics_TSVM_CUDA/PATCH_FP32.md")
_data = tim("/kaggle/input/**/Kvasir_YOLO_SEG_BG20/images/val")
assert _repo and _data, "❌ Chưa thêm dataset h4-tsvm-cuda (Add Input)"
CUSTOM_REPO_PATH = os.path.dirname(_repo[0])
DATASET_ROOT = os.path.dirname(os.path.dirname(_data[0]))
if not os.path.exists("/kaggle/working/ultralytics"):
    os.symlink(CUSTOM_REPO_PATH, "/kaggle/working/ultralytics")
if "/kaggle/working" not in sys.path:
    sys.path.insert(0, "/kaggle/working")
# "auto": tensor trên GPU dùng kernel; tensor CPU (chỉ lúc dựng mô hình, Ultralytics chạy thử 1 lần trên CPU để tính stride)
# dùng PyTorch. Lần chạy 25/11 đặt "cuda" nên bị lỗi "cần tensor trên GPU" ở bước dựng mô hình.
os.environ["TSVM_SCAN"] = "auto"
import ultralytics
import ultralytics.nn.modules.topology_shape_vmamba as tsv
assert ultralytics.__version__ == "8.4.127" and ultralytics.__file__.startswith("/kaggle/working/ultralytics")
assert tsv.cuda_kernel_available(), tsv._SELECTIVE_SCAN_IMPORT_ERROR
assert tsv.scan_backend() == "auto"
# Chặn an toàn: nếu nhánh PyTorch bị gọi với tensor GPU (tức là KHÔNG dùng kernel khi huấn luyện) thì báo lỗi ngay.
_goc_pt = tsv.SS2D._forward_pytorch
DEM_QUET = {"cpu_pytorch": 0, "gpu_kernel": 0}
def _pt_kiem(self, x):
    if x.is_cuda:
        raise RuntimeError("SS2D chạy nhánh PyTorch trên GPU: kernel không được dùng")
    DEM_QUET["cpu_pytorch"] += 1
    return _goc_pt(self, x)
_goc_cu = tsv.SS2D._forward_cuda
def _cu_dem(self, x):
    DEM_QUET["gpu_kernel"] += 1
    return _goc_cu(self, x)
tsv.SS2D._forward_pytorch = _pt_kiem
tsv.SS2D._forward_cuda = _cu_dem
print(f"✅ Ultralytics {ultralytics.__version__} (Hướng 4), TSVM_SCAN={tsv.scan_backend()}")

YAML_S = "/kaggle/working/yolo26s-seg-TopologyShapeVMamba.yaml"
with open(f"{CUSTOM_REPO_PATH}/cfg/models/26/yolo26-seg-TopologyShapeVMamba.yaml", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)
cfg["scale"] = "s"
with open(YAML_S, "w", encoding="utf-8") as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
dem = {s: len(os.listdir(f"{DATASET_ROOT}/images/{s}")) for s in ("train", "val")}
assert dem == DEM_DUNG, dem
DATA_YAML = "/kaggle/working/data_bg20.yaml"
with open(DATA_YAML, "w", encoding="utf-8") as f:
    yaml.safe_dump({"path": DATASET_ROOT, "train": "images/train", "val": "images/val", "nc": 1, "names": {0: "polyp"}}, f)
print("📁", DATASET_ROOT, dem)
'''.replace("DEM_DUNG", json.dumps(DEM))

SEEDS_GOC = [0, 15]   # 0: mốc so sánh chính; 15: lượt có mAP gần trung bình 20 seed, best.pt dùng cho web PolypVision
THAM_CHIEU = {0: {"mask_map50_95": 0.7245, "epoch": 88, "gio": 4.52},
              15: {"mask_map50_95": 0.7262, "epoch": 84, "gio": 4.73}}   # từ results.csv lượt báo cáo (PyTorch)

CELL2 = r"""# ============================================================
# CELL 2: HUẤN LUYỆN LẦN LƯỢT CÁC SEED TRONG SEEDS (siêu tham số chép nguyên từ args.yaml lượt báo cáo, chỉ đổi seed)
# Mỗi seed khoảng 2,4 giờ (đo ở mốc 3); 4 seed khoảng 9,6 giờ, dưới giới hạn 12 giờ của Kaggle.
# ============================================================
import random, numpy as np
from ultralytics import YOLO
ARGS = ARGS_BAO_CAO
SEEDS = SEEDS_DS
torch.use_deterministic_algorithms(True, warn_only=False)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
GIO = {}
for SEED in SEEDS:
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    TEN = f"Kvasir_BG20_YOLO26s_seg_TSVM_CUDA_s{SEED}_w{ARGS['workers']}"
    if os.path.exists(f"/kaggle/working/runs/{TEN}/weights/last.pt") and os.path.exists(f"/kaggle/working/runs/{TEN}/gio.txt"):
        print("Bỏ qua, đã xong:", TEN); continue
    model = YOLO(YAML_S).load("yolo26s-seg.pt")
    assert type(model.model.model[10]).__name__ == "C2TSVMamba"
    t0 = time.time()
    model.train(data=DATA_YAML, device=0, project="/kaggle/working/runs", name=TEN, exist_ok=True, **dict(ARGS, seed=SEED))
    GIO[SEED] = (time.time() - t0) / 3600
    open(f"/kaggle/working/runs/{TEN}/gio.txt", "w").write(f"{GIO[SEED]:.4f}")
    print(f"⏱️ seed {SEED}: {GIO[SEED]:.2f} giờ | số lần quét: {DEM_QUET}")
    del model; torch.cuda.empty_cache()
""".replace("ARGS_BAO_CAO", repr(args))

CELL3 = r"""# ============================================================
# CELL 3: TỔNG HỢP TỪNG SEED, SO VỚI LƯỢT BÁO CÁO CÙNG SEED (quy tắc |Δ| ≤ 0,0074 = 1 SD 20 seed), NÉN KẾT QUẢ
# ============================================================
import pandas as pd
THAM_CHIEU = THAM_CHIEU_DS
tong = []
for SEED in SEEDS:
    TEN = f"Kvasir_BG20_YOLO26s_seg_TSVM_CUDA_s{SEED}_w{ARGS['workers']}"
    RUN = f"/kaggle/working/runs/{TEN}"
    d = pd.read_csv(f"{RUN}/results.csv"); d.columns = [c.strip() for c in d.columns]
    i = d["metrics/mAP50-95(M)"].idxmax()
    t = {"seed": SEED, "gio_huan_luyen": float(open(f"{RUN}/gio.txt").read()), "gpu": torch.cuda.get_device_name(0),
         "torch": torch.__version__, "scan_backend": tsv.scan_backend(),
         "epoch_tot_nhat": int(d.loc[i, "epoch"]), "mask_map50_95_tot_nhat": float(d.loc[i, "metrics/mAP50-95(M)"]),
         "mask_map50_tai_do": float(d.loc[i, "metrics/mAP50(M)"]), "box_map50_95_tai_do": float(d.loc[i, "metrics/mAP50-95(B)"]),
         "tham_chieu_pytorch": THAM_CHIEU[SEED]}
    t["delta_map"] = round(t["mask_map50_95_tot_nhat"] - THAM_CHIEU[SEED]["mask_map50_95"], 4)
    t["ket_luan"] = "tương đương" if abs(t["delta_map"]) <= 0.0074 else "lệch hơn 1 SD"
    json.dump(t, open(f"{RUN}/tom_tat_moc3.json", "w"), ensure_ascii=False, indent=1)
    tong.append(t)
print(pd.DataFrame(tong)[["seed", "mask_map50_95_tot_nhat", "delta_map", "epoch_tot_nhat", "gio_huan_luyen", "ket_luan"]])
json.dump(tong, open("/kaggle/working/runs/tom_tat_moc3_tat_ca.json", "w"), ensure_ascii=False, indent=1)
shutil.make_archive("/kaggle/working/TEN_ZIP", "zip", "/kaggle/working/runs")
print("Tải về: TEN_ZIP.zip | tổng phiên", round((time.time() - T_PHIEN) / 3600, 2), "giờ")
"""


TM_RUNS = TM / "03_ket_qua_huan_luyen" / "TSVM_FP32"


def tham_chieu(seed):
    """Giá trị lượt báo cáo (PyTorch) cùng seed, tính từ results.csv gốc."""
    import pandas as pd
    d = pd.read_csv(TM_RUNS / f"Kvasir_BG20_YOLO26s_seg_TSVM_FP32_s{seed}_w2" / "results.csv")
    d.columns = [c.strip() for c in d.columns]
    i = d["metrics/mAP50-95(M)"].idxmax()
    r = {"mask_map50_95": round(float(d.loc[i, "metrics/mAP50-95(M)"]), 5), "epoch": int(d.loc[i, "epoch"])}
    if "time" in d.columns:
        r["gio"] = round(float(d["time"].iloc[-1]) / 3600, 2)
    return r


# Nhóm chạy: [0, 15] đã xong; 18 seed còn lại chia 1 notebook 2 seed + 4 notebook 4 seed
NHOM = [[0, 15], [1, 2], [3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14], [16, 17, 18, 19]]


def tao(seeds):
    ten = "s" + "-s".join(map(str, seeds))
    zip_ = "h4_moc3_" + ten.replace("-", "_")
    tc = {s: tham_chieu(s) for s in seeds}
    c0 = CELL0.replace("HUẤN LUYỆN TSVM SEED 0 BẰNG", f"HUẤN LUYỆN TSVM SEED {', '.join(map(str, seeds))} BẰNG")
    c2 = CELL2.replace("SEEDS_DS", repr(seeds))
    c3 = CELL3.replace("THAM_CHIEU_DS", repr(tc)).replace("TEN_ZIP", zip_)
    p = OUT / "3_notebooks" / f"h4-moc3-huan-luyen-{ten}.ipynb"
    p.write_text(json.dumps(nb([c0, CELL1, c2, c3]), ensure_ascii=False, indent=1), encoding="utf-8")
    print("Notebook:", p.name, "| seed", seeds, "| zip", zip_ + ".zip")
    for s in seeds:
        print("   tham chiếu", s, tc[s])


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    assert sorted(sum(NHOM, [])) == list(range(20))
    for g in NHOM:
        tao(g)
    print("Số siêu tham số chép:", len(args))


if __name__ == "__main__":
    main()
