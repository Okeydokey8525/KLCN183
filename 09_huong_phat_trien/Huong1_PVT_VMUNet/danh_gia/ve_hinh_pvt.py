"""Vẽ hình cho slide: so sánh Polyp-PVT 640 với 4 mô hình của báo cáo (seed 0–4, đánh giá theo pixel).

Đọc: ket_qua/danh_gia_pixel/{11_pixel_tung_seed_5model.csv, 10_pixel_tung_anh_polyppvt.csv, du_lieu_dinh_tinh/}
      và (chỉ đọc) Test_Mau/.../pixel_640/{10_pixel_tung_anh.csv, du_lieu_dinh_tinh/}.
Ghi: ket_qua/danh_gia_pixel/hinh/<tên>.png (300 dpi) và .pdf.
Phong cách (màu, cỡ chữ, dấu phẩy thập phân) giống 06_bao_cao/hinh/scripts/ve_hinh.py của báo cáo.

Chạy:  C:\\venv_kl_gpu\\Scripts\\python.exe ve_hinh_pvt.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
GOC = HERE.parent
KQ = GOC / "ket_qua" / "danh_gia_pixel"
OUT = KQ / "hinh"
TEST_MAU = GOC.parent.parent / "Test_Mau"
DATA = TEST_MAU / "01_du_lieu" / "Kvasir_YOLO_SEG_BG20"
PIX_CU = TEST_MAU / "00_tai_lieu_tong_quan" / "bang" / "pixel_640"

CM = 1 / 2.54
W = 16.0 * CM          # rộng như hình trong báo cáo
FONT = 10
C_GRID = "#D9D9D9"
# (khóa, màu, nhãn). Màu 4 mô hình cũ giữ đúng như báo cáo; Polyp-PVT thêm màu chàm.
MODELS = [("baseline", "#0072B2", "YOLO26s-seg"),
          ("tsvm", "#D55E00", "YOLO26s-seg + TSVM"),
          ("pranet640", "#009E73", "PraNet (ResNet-50)"),
          ("polyppvt640", "#332288", "Polyp-PVT (PVTv2-B2)"),
          ("unet640", "#CC79A7", "U-Net")]
NHOM = {"baseline": "YOLO", "tsvm": "YOLO + Mamba", "pranet640": "CNN", "polyppvt640": "Transformer", "unet640": "CNN"}


def setup():
    plt.rcParams.update({
        "font.size": FONT, "axes.titlesize": FONT, "axes.labelsize": FONT,
        "xtick.labelsize": FONT - 1, "ytick.labelsize": FONT - 1, "legend.fontsize": FONT - 1,
        "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "lines.linewidth": 1.2,
        "legend.frameon": False, "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def num(v, nd):
    return f"{v:.{nd}f}".replace(".", ",")


def num_axis(ax, nd):
    from matplotlib.ticker import FuncFormatter
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _p: num(v, nd)))


def grid_y(ax):
    ax.yaxis.grid(True, color=C_GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    fig.savefig(OUT / f"{name}.pdf")
    plt.close(fig)
    print("  ", OUT / f"{name}.png")


def nhan_xuong_dong(lab):
    return lab.replace(" + ", "\n+ ").replace(" (", "\n(")


# ---------------------------------------------------------------- h01: cột Dice/IoU/P/R
def h01_cot_pixel(sd):
    metrics = [("dice", "Dice"), ("iou", "IoU"), ("precision", "Precision"), ("recall", "Recall")]
    fig, ax = plt.subplots(figsize=(W, W * 0.55))
    x = np.arange(len(metrics))
    n = len(MODELS)
    w = 0.16
    for k, (key, c, lab) in enumerate(MODELS):
        g = sd[sd.model == key]
        mu = [g[m].mean() for m, _ in metrics]
        s_ = [g[m].std(ddof=1) for m, _ in metrics]
        xs = x + (k - (n - 1) / 2) * w
        ax.bar(xs, mu, w * 0.92, color=c, alpha=0.88, label=lab, zorder=2,
               edgecolor="black" if key == "polyppvt640" else "none", linewidth=0.6)
        ax.errorbar(xs, mu, yerr=s_, fmt="none", ecolor="black", elinewidth=0.7, capsize=1.6, zorder=3)
    ax.set_xticks(x, [lab for _, lab in metrics])
    ax.set_ylabel("Giá trị theo pixel")
    ax.set_ylim(0.5, 1.0)
    num_axis(ax, 1)
    grid_y(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2, columnspacing=1.0)
    ax.text(1.0, -0.13, "mean ± SD, seed 0–4; 120 ảnh polyp của tập val; trục tung bắt đầu từ 0,5",
            transform=ax.transAxes, ha="right", va="top", fontsize=FONT - 3, color="#555555")
    save(fig, "h01_cot_pixel_5_mo_hinh")


# ---------------------------------------------------------------- h02: Dice từng seed, nối cùng seed
def h02_dice_tung_seed(sd):
    keys = [m for m in MODELS if m[0] != "unet640"]   # U-Net thấp hơn hẳn (~0,70), bỏ để giữ thang đo đọc được
    fig, ax = plt.subplots(figsize=(W * 0.8, W * 0.55))
    piv = sd.pivot(index="seed", columns="model", values="dice")
    xs = np.arange(len(keys))
    for s in piv.index:
        ax.plot(xs, [piv.loc[s, k] for k, _, _ in keys], color="#BBBBBB", linewidth=0.7, zorder=1)
    for i, (key, c, lab) in enumerate(keys):
        g = piv[key]
        ax.scatter(np.full(len(g), xs[i]), g, s=22, color=c, edgecolors="white", linewidths=0.4, zorder=3)
        ax.hlines(g.mean(), xs[i] - 0.22, xs[i] + 0.22, color="black", linewidth=1.4, zorder=4)
        ax.text(xs[i] + 0.26, g.mean(), num(g.mean(), 3), va="center", fontsize=FONT - 2)
    ax.set_xticks(xs, [nhan_xuong_dong(lab) for _, _, lab in keys], fontsize=FONT - 2)
    ax.set_xlim(-0.4, len(keys) - 0.4)
    ax.set_ylabel("Dice theo pixel (trung bình 120 ảnh)")
    num_axis(ax, 2)
    grid_y(ax)
    ax.text(1.0, 1.02, "đường xám nối cùng một seed; vạch đen là trung bình 5 seed",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=FONT - 3, color="#555555")
    save(fig, "h02_dice_tung_seed")


# ---------------------------------------------------------------- h03: ảnh minh họa seed 0
def _mask(npz, name, h, w):
    return np.unpackbits(npz[name])[: h * w].reshape(h, w).astype(bool)


def h03_minh_hoa():
    cu = pd.read_csv(PIX_CU / "10_pixel_tung_anh.csv", encoding="utf-8-sig")
    moi = pd.read_csv(KQ / "10_pixel_tung_anh_polyppvt.csv", encoding="utf-8-sig")
    pix = pd.concat([cu, moi], ignore_index=True)
    pol = pix[(pix.seed == 0) & ~pix.is_bg]
    dice = pol.assign(d=2 * pol.tp / (2 * pol.tp + pol.fp + pol.fn)).pivot(index="image", columns="model", values="d")
    keys = [m for m in MODELS if m[0] != "unet640"]
    # Chọn ảnh theo Dice trung bình của cả 4 mô hình (không chọn theo chênh lệch giữa hai mô hình,
    # để tránh thiên về một bên): dễ nhất, trung vị, khó nhất.
    tb = dice[[k for k, _, _ in keys]].mean(axis=1).sort_values()
    picks = [tb.index[-1], tb.index[len(tb) // 2], tb.index[0]]
    npz = {k: np.load((KQ if k == "polyppvt640" else PIX_CU) / "du_lieu_dinh_tinh" / f"pred_seed0_{k}.npz")
           for k, _, _ in keys}
    cols = ["Ảnh gốc", "Nhãn", "YOLO26s-seg", "TSVM", "PraNet", "Polyp-PVT"]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(W, W * 0.52))
    for r, name in enumerate(picks):
        img = cv2.cvtColor(cv2.imread(str(DATA / "images" / "val" / name)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        gt = np.zeros((h, w), np.uint8)
        for line in (DATA / "labels" / "val" / f"{Path(name).stem}.txt").read_text().splitlines():
            c = np.array(line.split()[1:], np.float32).reshape(-1, 2) * [w, h]
            cv2.fillPoly(gt, [c.astype(np.int32)], 1)
        panels = [img, gt.astype(bool)] + [_mask(npz[k], name, h, w) for k, _, _ in keys]
        for ci, (ax, pnl) in enumerate(zip(axes[r], panels)):
            if ci == 0:
                ax.imshow(pnl)
            else:
                over = img.astype(float) / 255
                color = np.array(mcolors.to_rgb("#FFFFFF" if ci == 1 else keys[ci - 2][1]))
                over[pnl] = 0.45 * over[pnl] + 0.55 * color
                ax.imshow(over)
                ax.contour(gt, levels=[0.5], colors="yellow", linewidths=0.6)
                if ci >= 2:
                    ax.text(0.03, 0.04, f"Dice {num(dice.loc[name, keys[ci - 2][0]], 2)}", transform=ax.transAxes,
                            fontsize=FONT - 3, color="white", bbox=dict(facecolor="black", alpha=0.55, pad=1, linewidth=0))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[ci], fontsize=FONT - 2)
    fig.subplots_adjust(wspace=0.03, hspace=0.05)
    save(fig, "h03_minh_hoa_seed0")
    return picks


def main():
    setup()
    sd = pd.read_csv(KQ / "11_pixel_tung_seed_5model.csv", encoding="utf-8-sig")
    assert set(sd.model) == {k for k, _, _ in MODELS} and sorted(sd.seed.unique()) == [0, 1, 2, 3, 4]
    h01_cot_pixel(sd)
    h02_dice_tung_seed(sd)
    print("   ảnh minh họa:", h03_minh_hoa())


if __name__ == "__main__":
    main()
