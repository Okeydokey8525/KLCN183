"""Chi phí của 5 mô hình (seed 0–4): thời gian huấn luyện, số tham số, FLOPs, độ trễ GPU, VRAM khi suy luận.

Phần 1. Thời gian huấn luyện: đọc từ log có sẵn, không chạy lại gì.
  - YOLO26s-seg, TSVM: cột `time` (giây cộng dồn) ở dòng cuối results.csv của Ultralytics.
  - PraNet, U-Net, Polyp-PVT: cột `time_s` (giây cộng dồn) ở dòng cuối results.csv của notebook.
    Polyp-PVT seed 0 bị ngắt rồi chạy tiếp; cột này không bị đặt lại (script kiểm tra tăng đều).
Phần 2. Đo trên laptop, cùng một phiên cho cả 5 mô hình. Cách đo chép từ Test_Mau/.../do_flops_gpu.py:
  checkpoint seed 0, FP32, batch 1, ảnh 1×3×640×640, cudnn.benchmark tắt, chỉ tính forward
  (không tiền/hậu xử lý), khởi động 50 lần, đo 200 lần bằng CUDA event, 5 vòng, đảo thứ tự mỗi vòng,
  chờ GPU dưới 55 °C trước mỗi khối. YOLO được gộp Conv-BN như khi suy luận.
  FLOPs: torch.utils.flop_counter.FlopCounterMode (conv, matmul, bmm, einsum). Phép theo phần tử
  không được đếm, nên số của TSVM là cận dưới.
  Độ trễ tuyệt đối thay đổi giữa các phiên (xem 09_HIEU_NANG_TINH_TOAN.md), nên đo lại cả 5 mô hình
  trong cùng phiên và ưu tiên đọc tỉ lệ so với YOLO26s-seg.

Ghi vào ket_qua/chi_phi/. Test_Mau chỉ đọc.

    C:\\venv_kl_gpu\\Scripts\\python.exe do_chi_phi_5_mo_hinh.py [--chi-thoi-gian]
"""
from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GOC = HERE.parent
TEST_MAU = GOC.parent.parent / "Test_Mau"
RUNS = TEST_MAU / "03_ket_qua_huan_luyen"
OTHER = TEST_MAU / "05_so_sanh_mo_hinh_khac"
PVT_RUNS = GOC / "ket_qua" / "PolypPVT_640"
OUT = GOC / "ket_qua" / "chi_phi"
SEEDS = range(5)

THU_MUC = {
    "baseline": lambda s: RUNS / "Baseline_YOLO26s_seg" / f"Kvasir_BG20_Baseline_YOLO26s_seg_s{s}_w2",
    "tsvm": lambda s: RUNS / "TSVM_FP32" / f"Kvasir_BG20_YOLO26s_seg_TSVM_FP32_s{s}_w2",
    "pranet640": lambda s: OTHER / "PraNet_640" / f"PraNet_640_aug_s{s}",
    "unet640": lambda s: OTHER / "UNet_640" / f"UNet_640_aug_s{s}",
    "polyppvt640": lambda s: PVT_RUNS / f"PolypPVT_640_aug_s{s}",
}
CKPT = {
    "baseline": lambda s: THU_MUC["baseline"](s) / "weights" / "best.pt",
    "tsvm": lambda s: THU_MUC["tsvm"](s) / "weights" / "best.pt",
    "pranet640": lambda s: THU_MUC["pranet640"](s) / f"pranet_best_s{s}.pth",
    "unet640": lambda s: THU_MUC["unet640"](s) / f"best_unet_s{s}.pth",
    "polyppvt640": lambda s: THU_MUC["polyppvt640"](s) / f"polyp_pvt_best_s{s}.pth",
}
WARMUP, ITERS, ROUNDS = 50, 200, 5
COOL_C, COOL_MAX_S = 55, 180


# ---------------------------------------------------------------- phần 1: thời gian huấn luyện
def thoi_gian_huan_luyen():
    rows = []
    for key, d in THU_MUC.items():
        for s in SEEDS:
            r = pd.read_csv(d(s) / "results.csv")
            r.columns = [c.strip() for c in r.columns]
            # cả hai loại log đều ghi giây cộng dồn; kiểm tra không bị đặt lại khi chạy tiếp
            cot = "time" if key in ("baseline", "tsvm") else "time_s"
            assert (r[cot].diff().dropna() > 0).all(), (key, s)
            giay = float(r[cot].iloc[-1])
            row = {"model": key, "seed": s, "so_epoch": len(r), "gio_huan_luyen": giay / 3600}
            tt = d(s) / "tom_tat.txt"
            if tt.exists():
                m = re.search(r"train_time_s=([\d.]+)", tt.read_text(encoding="utf-8"))
                row["gio_tom_tat"] = float(m.group(1)) / 3600 if m else np.nan
            if "gpu_mem_gb" in r.columns:
                row["vram_huan_luyen_gb_max"] = float(r["gpu_mem_gb"].max())
            rows.append(row)
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "20_thoi_gian_huan_luyen_tung_seed.csv", index=False, encoding="utf-8-sig")
    tom = df.groupby("model", sort=False).agg(so_epoch=("so_epoch", "first"), gio_mean=("gio_huan_luyen", "mean"),
                                               gio_sd=("gio_huan_luyen", "std"), gio_min=("gio_huan_luyen", "min"),
                                               gio_max=("gio_huan_luyen", "max"))
    print(tom.round(2).to_string())
    return tom


# ---------------------------------------------------------------- phần 2: đo trên laptop
def gpu_state():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.split(",")
        return float(out[0]), float(out[1])
    except Exception:
        return np.nan, np.nan


def cho_nguoi():
    t0 = time.time()
    while time.time() - t0 < COOL_MAX_S and not gpu_state()[0] < COOL_C:
        time.sleep(5)
    return time.time() - t0


def dung_mo_hinh():
    """Trả về {khóa: (mạng nn.Module FP32 eval trên CPU, số tham số)} của checkpoint seed 0."""
    import torch
    sys.path.insert(0, str(TEST_MAU / "00_tai_lieu_tong_quan" / "scripts"))
    sys.path.insert(0, str(HERE))
    import danh_gia_pixel as dg            # chỉ đọc: hàm dựng PraNet/U-Net và fork TSVM
    import danh_gia_pixel_pvt as dp
    cleanup = dg.import_fork()
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER
    LOGGER.setLevel("ERROR")
    nets = {}
    for key in ("baseline", "tsvm"):
        yolo = YOLO(str(CKPT[key](0)))
        n = sum(p.numel() for p in yolo.model.parameters())
        nets[key] = (yolo.model.fuse().float().eval(), n)
    cls = dg.notebook_classes()
    for key in ("pranet640", "unet640"):
        net = cls[key]()
        net.load_state_dict(torch.load(CKPT[key](0), map_location="cpu", weights_only=True), strict=True)
        nets[key] = (net.float().eval(), sum(p.numel() for p in net.parameters()))
    net = dp.tao_polyp_pvt()
    net.load_state_dict(torch.load(CKPT["polyppvt640"](0), map_location="cpu", weights_only=True), strict=True)
    nets["polyppvt640"] = (net.float().eval(), sum(p.numel() for p in net.parameters()))
    return nets, cleanup


def do_tren_laptop():
    import torch
    from torch.utils.flop_counter import FlopCounterMode
    assert torch.cuda.is_available(), "Cần GPU (C:\\venv_kl_gpu\\Scripts\\python.exe)"
    torch.backends.cudnn.benchmark = False
    nets, cleanup = dung_mo_hinh()
    try:
        info = []
        x_cpu = torch.zeros(1, 3, 640, 640)
        for key, (net, n) in nets.items():
            with torch.inference_mode(), FlopCounterMode(display=False) as fc:
                net(x_cpu)
            net.cuda()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            with torch.inference_mode():
                net(x_cpu.cuda())
            torch.cuda.synchronize()
            info.append({"model": key, "params": n, "gflops_flopcounter": fc.get_total_flops() / 1e9,
                         "vram_suy_luan_mb": torch.cuda.max_memory_allocated() / 2**20,
                         "ckpt_mib": CKPT[key](0).stat().st_size / 2**20})
            print(f"{key}: {n:,} tham số, {fc.get_total_flops() / 1e9:.2f} GFLOPs, "
                  f"VRAM {info[-1]['vram_suy_luan_mb']:.0f} MB", flush=True)

        x = torch.zeros(1, 3, 640, 640, device="cuda")
        rows, blocks = [], []
        thu_tu = list(nets)
        with torch.inference_mode():
            for r in range(ROUNDS):
                for key in (thu_tu if r % 2 == 0 else thu_tu[::-1]):
                    net = nets[key][0]
                    waited = cho_nguoi()
                    temp0, clk0 = gpu_state()
                    for _ in range(WARMUP):
                        net(x)
                    torch.cuda.synchronize()
                    for i in range(ITERS):
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        net(x)
                        e.record()
                        torch.cuda.synchronize()
                        rows.append({"model": key, "round": r, "iteration": i, "latency_ms": s.elapsed_time(e)})
                    temp1, clk1 = gpu_state()
                    med = np.median([q["latency_ms"] for q in rows[-ITERS:]])
                    blocks.append({"model": key, "round": r, "cho_nguoi_s": round(waited), "nhiet_truoc": temp0,
                                   "nhiet_sau": temp1, "xung_truoc_mhz": clk0, "xung_sau_mhz": clk1, "median_ms": med})
                    print(f"vòng {r} {key}: chờ {waited:.0f}s, {temp0:.0f}→{temp1:.0f}°C, "
                          f"xung {clk0:.0f}→{clk1:.0f} MHz, trung vị {med:.2f} ms", flush=True)
        lat = pd.DataFrame(rows)
        bl = pd.DataFrame(blocks)
        lat.to_csv(OUT / "21b_do_tre_gpu_tung_lan.csv", index=False, encoding="utf-8-sig")
        bl.to_csv(OUT / "21c_do_tre_gpu_nhiet_do.csv", index=False, encoding="utf-8-sig")
        g = lat.groupby("model").latency_ms
        out = pd.DataFrame(info).set_index("model")
        out["gpu_ms_median"], out["gpu_ms_min"], out["gpu_ms_p10"] = g.median(), g.min(), g.quantile(0.10)
        out["gpu_ms_mean"], out["gpu_ms_sd"] = g.mean(), g.std(ddof=1)
        # tỉ lệ so với YOLO26s-seg theo từng vòng (trung vị khối), báo khoảng min–max
        piv = bl.pivot(index="round", columns="model", values="median_ms")
        ratio = piv.div(piv["baseline"], axis=0)
        out["ti_le_vs_yolo_median"] = out["gpu_ms_median"] / out.loc["baseline", "gpu_ms_median"]
        out["ti_le_vs_yolo_vong_min"], out["ti_le_vs_yolo_vong_max"] = ratio.min(), ratio.max()
        out["gpu"] = torch.cuda.get_device_name(0)
        out["torch"] = torch.__version__
        out["platform"] = platform.platform()
        out.reset_index().to_csv(OUT / "21_flops_do_tre_gpu_5model.csv", index=False, encoding="utf-8-sig")
        with pd.option_context("display.width", 250, "display.float_format", "{:.3f}".format):
            print(out.drop(columns=["platform", "gpu", "torch"]).to_string())
    finally:
        cleanup()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi-thoi-gian", action="store_true", help="chỉ đọc thời gian huấn luyện, không đo GPU")
    a = ap.parse_args()
    thoi_gian_huan_luyen()
    if not a.chi_thoi_gian:
        do_tren_laptop()


if __name__ == "__main__":
    main()
