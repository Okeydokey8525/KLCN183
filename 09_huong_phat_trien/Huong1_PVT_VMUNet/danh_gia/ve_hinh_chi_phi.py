"""Vẽ hình chi phí của 5 mô hình (cho slide). Phong cách lấy từ ve_hinh_pvt.py.

Đọc: ket_qua/chi_phi/{20_thoi_gian_huan_luyen_tung_seed.csv, 21_flops_do_tre_gpu_5model.csv}
      ket_qua/danh_gia_pixel/11_pixel_tung_seed_5model.csv
Ghi: ket_qua/chi_phi/hinh/<tên>.png (300 dpi) và .pdf

    C:\\venv_kl_gpu\\Scripts\\python.exe ve_hinh_chi_phi.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import ve_hinh_pvt as v

KQ = v.GOC / "ket_qua"
CP = KQ / "chi_phi"
OUT = CP / "hinh"
NHAN_NGAN = {"baseline": "YOLO26s-seg", "tsvm": "YOLO + TSVM", "pranet640": "PraNet",
             "polyppvt640": "Polyp-PVT", "unet640": "U-Net"}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("  ", OUT / f"{name}.png")


def doc():
    cp = pd.read_csv(CP / "21_flops_do_tre_gpu_5model.csv", encoding="utf-8-sig").set_index("model")
    tg = pd.read_csv(CP / "20_thoi_gian_huan_luyen_tung_seed.csv", encoding="utf-8-sig")
    sd = pd.read_csv(KQ / "danh_gia_pixel" / "11_pixel_tung_seed_5model.csv", encoding="utf-8-sig")
    cp["ratio_p10"] = cp.gpu_ms_p10 / cp.loc["baseline", "gpu_ms_p10"]
    cp["gio_mean"] = tg.groupby("model").gio_huan_luyen.mean()
    cp["gio_sd"] = tg.groupby("model").gio_huan_luyen.std(ddof=1)
    cp["dice_mean"] = sd.groupby("model").dice.mean()
    cp["dice_sd"] = sd.groupby("model").dice.std(ddof=1)
    return cp


# ---------------------------------------------------------------- c01: 6 ô cột chi phí
def c01_bang_chi_phi(cp):
    o = [k for k, _, _ in v.MODELS]
    mau = [c for _, c, _ in v.MODELS]
    panels = [
        ("Dice theo pixel (seed 0–4)", cp.dice_mean, cp.dice_sd, 3, (0.6, 0.97)),
        ("Số tham số (triệu)", cp.params / 1e6, None, 1, None),
        ("GFLOPs (ảnh 640×640)", cp.gflops_flopcounter, None, 0, None),
        ("Huấn luyện (giờ/seed)", cp.gio_mean, cp.gio_sd, 1, None),
        ("Độ trễ GPU (× YOLO26s-seg)", cp.ratio_p10, None, 2, None),
        ("VRAM suy luận (MB)", cp.vram_suy_luan_mb, None, 0, None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(v.W, v.W * 0.62))
    for ax, (tit, val, err, nd, ylim) in zip(axes.flat, panels):
        y = val.reindex(o).values
        x = np.arange(len(o))
        for xi, yi, k, c in zip(x, y, o, mau):
            ax.bar(xi, yi, 0.72, color=c, alpha=0.88, zorder=2,
                   edgecolor="black" if k == "polyppvt640" else c, linewidth=0.6 if k == "polyppvt640" else 0)
        top = y + (err.reindex(o).values if err is not None else 0)
        if err is not None:
            ax.errorbar(x, y, yerr=err.reindex(o).values, fmt="none", ecolor="black", elinewidth=0.7,
                        capsize=1.6, zorder=3)
        lo, hi = ylim if ylim else (0, y.max() * 1.18)
        ax.set_ylim(lo, hi)
        for xi, yi, ti in zip(x, y, top):
            ax.text(xi, ti + (hi - lo) * 0.02, v.num(yi, nd), ha="center", va="bottom", fontsize=v.FONT - 3)
        ax.set_title(tit, fontsize=v.FONT - 1)
        ax.set_xticks(x, [NHAN_NGAN[k] for k in o], rotation=35, ha="right", fontsize=v.FONT - 3)
        ax.tick_params(axis="y", labelsize=v.FONT - 3)
        v.num_axis(ax, 2 if nd >= 2 else 0)
        v.grid_y(ax)
    fig.text(1.0, -0.01, "Dice: mean ± SD seed 0–4, trục từ 0,6. Tham số, FLOPs, độ trễ, VRAM: checkpoint seed 0, "
             "RTX 5050 Laptop, FP32. Độ trễ: phân vị 10%. Huấn luyện: log Kaggle, mean ± SD seed 0–4. "
             "FLOPs của TSVM là cận dưới.", ha="right", va="top", fontsize=v.FONT - 4, color="#555555", wrap=True)
    fig.tight_layout(h_pad=1.0, w_pad=0.8)
    save(fig, "c01_chi_phi_5_mo_hinh")


# ---------------------------------------------------------------- c02: Dice so với độ trễ
def c02_dice_va_do_tre(cp):
    fig, ax = plt.subplots(figsize=(v.W * 0.8, v.W * 0.5))
    for k, c, lab in v.MODELS:
        r = cp.loc[k]
        s = 25 + r.params / 1e6 * 9
        ax.errorbar(r.ratio_p10, r.dice_mean, yerr=r.dice_sd, fmt="none", ecolor=c, elinewidth=0.8, capsize=2, zorder=2)
        ax.scatter(r.ratio_p10, r.dice_mean, s=s, color=c, alpha=0.85, zorder=3,
                   edgecolors="black" if k == "polyppvt640" else "white", linewidths=0.6)
        dx, dy, ha = (0.08, 0.0, "left")
        if k == "tsvm":
            dy = -0.022
        if k == "baseline":
            dx, dy = 0.0, -0.035
            ha = "center"
        ax.annotate(f"{NHAN_NGAN[k]}\n{v.num(r.params / 1e6, 1)} M tham số", (r.ratio_p10, r.dice_mean),
                    xytext=(r.ratio_p10 + dx, r.dice_mean + dy), fontsize=v.FONT - 3, ha=ha, va="center")
    ax.set_xlabel("Độ trễ GPU so với YOLO26s-seg (lần, phân vị 10%)")
    ax.set_ylabel("Dice theo pixel")
    ax.set_xlim(0.7, 3.6)
    ax.set_ylim(0.65, 0.93)
    v.num_axis(ax, 2)
    from matplotlib.ticker import FuncFormatter
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _p: v.num(x, 1)))
    v.grid_y(ax)
    ax.text(1.0, 1.02, "cỡ chấm theo số tham số; thanh dọc: SD qua seed 0–4; góc trên bên trái là tốt",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=v.FONT - 3, color="#555555")
    save(fig, "c02_dice_va_do_tre")


# ---------------------------------------------------------------- c03: thời gian huấn luyện từng seed
def c03_huan_luyen(cp):
    tg = pd.read_csv(CP / "20_thoi_gian_huan_luyen_tung_seed.csv", encoding="utf-8-sig")
    fig, ax = plt.subplots(figsize=(v.W * 0.8, v.W * 0.45))
    o = [m for m in v.MODELS]
    for i, (k, c, lab) in enumerate(o):
        g = tg[tg.model == k].gio_huan_luyen
        ax.barh(i, g.mean(), 0.6, color=c, alpha=0.88, zorder=2,
                edgecolor="black" if k == "polyppvt640" else "none", linewidth=0.6)
        ax.scatter(g, np.full(len(g), i), s=10, color="black", zorder=3)
        ax.text(g.max() + 0.25, i, f"{v.num(g.mean(), 1)} giờ  (×{v.num(g.mean() / cp.loc['baseline', 'gio_mean'], 1)})",
                va="center", fontsize=v.FONT - 2)
    ax.set_yticks(range(len(o)), [NHAN_NGAN[k] for k, _, _ in o])
    ax.invert_yaxis()
    ax.set_xlabel("Giờ huấn luyện mỗi seed (100 epoch, batch 8, ảnh 640, GPU Kaggle)")
    ax.set_xlim(0, 18)
    ax.xaxis.grid(True, color=v.C_GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.text(1.0, 1.02, "chấm đen: từng seed 0–4; (×) so với YOLO26s-seg", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=v.FONT - 3, color="#555555")
    save(fig, "c03_thoi_gian_huan_luyen")


def main():
    v.setup()
    cp = doc()
    assert set(cp.index) == {k for k, _, _ in v.MODELS}
    print(cp[["params", "gflops_flopcounter", "ratio_p10", "gio_mean", "dice_mean"]].round(3).to_string())
    c01_bang_chi_phi(cp)
    c02_dice_va_do_tre(cp)
    c03_huan_luyen(cp)


if __name__ == "__main__":
    main()
