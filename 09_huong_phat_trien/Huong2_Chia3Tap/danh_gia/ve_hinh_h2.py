"""Vẽ hình hướng 2 từ ../ket_qua/danh_gia/bang/ (không suy luận). Ghi ../ket_qua/danh_gia/hinh/<tên>.png (300 dpi) và .pdf.

    python ve_hinh_h2.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
H2 = HERE.parent
KQ = H2 / "ket_qua"
BANG = KQ / "danh_gia" / "bang"
HINH = KQ / "danh_gia" / "hinh"
DATA = H2 / "du_lieu" / "Kvasir_BG20_3tap"
C = {"yolo": "#0072B2", "tsvm": "#D55E00"}
LBL = {"yolo": "YOLO26s-seg", "tsvm": "YOLO26s-seg + TSVM"}
RUN = {"yolo": lambda s: KQ / "YOLO26s_seg" / f"H2_BG20_3tap_YOLO26s_seg_s{s}",
       "tsvm": lambda s: KQ / "TSVM_FP32" / f"H2_BG20_3tap_TSVM_FP32_s{s}"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    HINH.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(HINH / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ", HINH / f"{name}.png")


def paired_plot(sd, metric, ylabel, name, title):
    fig, ax = plt.subplots(figsize=(5.2, 4))
    y = sd[sd.model == "yolo"].set_index("seed")[metric]
    t = sd[sd.model == "tsvm"].set_index("seed")[metric]
    for s in y.index:
        ax.plot([0, 1], [y[s], t[s]], color="#999999", lw=0.8, zorder=1)
    for x, key, v in [(0, "yolo", y), (1, "tsvm", t)]:
        ax.scatter(np.full(len(v), x), v, color=C[key], s=28, zorder=2)
        ax.errorbar(x + (-0.12 if x == 0 else 0.12), v.mean(), yerr=v.std(ddof=1), fmt="o", color=C[key],
                    capsize=4, ms=7, zorder=3)
        ax.text(x + (-0.18 if x == 0 else 0.18), v.mean(), f"{v.mean():.3f}\n±{v.std(ddof=1):.3f}",
                ha="right" if x == 0 else "left", va="center", fontsize=9, color=C[key])
    ax.set_xticks([0, 1], [LBL["yolo"], LBL["tsvm"]])
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    save(fig, name)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sd = pd.read_csv(BANG / "11_tung_seed.csv", encoding="utf-8-sig")
    st = pd.read_csv(BANG / "12_thong_ke.csv", encoding="utf-8-sig")
    old = pd.read_csv(BANG / "13_bao_cao_cu_seed0_9.csv", encoding="utf-8-sig")

    def note(metric):
        r = st[(st.so_sanh == "H2: TSVM - YOLO") & (st.metric == metric)].iloc[0]
        return f"Δ = {r.delta:+.4f}, p t = {r.paired_t_p:.2f}, p Wilcoxon = {r.wilcoxon_p:.2f}, TSVM cao hơn {r.a_higher}/10 seed"

    paired_plot(sd, "dice", "Dice theo pixel (120 ảnh polyp)", "h01_dice_tung_seed_test",
                f"Hướng 2, tập test, 10 seed ghép cặp\n{note('dice')}")
    paired_plot(sd, "map50_95_mask", "mask mAP50–95 (tập test)", "h02_map_tung_seed_test",
                f"Hướng 2, tập test, 10 seed ghép cặp\n{note('map50_95_mask')}")

    # h03: H2 (tách val) so với báo cáo cũ (chọn checkpoint trên chính tập này), cùng 160 ảnh, seed 0–9
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (m, lab) in zip(axes, [("dice", "Dice"), ("iou", "IoU"), ("bg_fp_image_rate", "Tỉ lệ ảnh nền bị báo nhầm")]):
        for i, key in enumerate(["yolo", "tsvm"]):
            for j, (src, name, hatch) in enumerate([(old, "Báo cáo cũ", "//"), (sd, "Hướng 2", None)]):
                v = src[src.model == key][m]
                x = i * 2.6 + j
                ax.bar(x, v.mean(), yerr=v.std(ddof=1), color=C[key], alpha=0.45 if j == 0 else 0.95,
                       hatch=hatch, edgecolor="white", capsize=3, width=0.85)
                ax.text(x, v.mean() + v.std(ddof=1) + (0.004 if m != "bg_fp_image_rate" else 0.002),
                        f"{v.mean():.3f}", ha="center", fontsize=8)
        ax.set_xticks([0, 1, 2.6, 3.6], ["cũ", "H2", "cũ", "H2"])
        ax.set_title(lab, fontsize=10)
        if m != "bg_fp_image_rate":
            ax.set_ylim(0.7, 0.9)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(color=C["yolo"], label=LBL["yolo"]), Patch(color=C["tsvm"], label=LBL["tsvm"]),
                        Patch(facecolor="#bbbbbb", hatch="//", edgecolor="white", label="Báo cáo cũ: chọn checkpoint trên chính tập này"),
                        Patch(facecolor="#555555", label="Hướng 2: chọn trên val riêng, báo cáo trên test")],
               loc="upper center", ncol=2, fontsize=8, frameon=False, bbox_to_anchor=(0.5, 1.12))
    fig.text(0.5, -0.04, "Cùng 160 ảnh (120 polyp + 40 nền), seed 0–9, mean ± SD qua seed", ha="center", fontsize=8)
    save(fig, "h03_so_voi_bao_cao_cu")

    # h04: đường cong val (mask mAP50–95 theo epoch), mean và min–max qua 10 seed
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key in ["yolo", "tsvm"]:
        cur = []
        for s in range(10):
            r = pd.read_csv(RUN[key](s) / "results.csv")
            r.columns = [c.strip() for c in r.columns]
            cur.append(r["metrics/mAP50-95(M)"].to_numpy())
        a = np.vstack(cur)
        ep = np.arange(1, a.shape[1] + 1)
        ax.plot(ep, a.mean(0), color=C[key], label=LBL[key])
        ax.fill_between(ep, a.min(0), a.max(0), color=C[key], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mask mAP50–95 trên val mới")
    ax.set_title("Đường cong val (trung bình, vùng min–max qua 10 seed)", fontsize=10)
    ax.legend(frameon=False)
    save(fig, "h04_duong_cong_val")

    # h05: minh họa seed 0, 4 ảnh test: ảnh, nhãn, YOLO, TSVM
    ta = pd.read_csv(BANG / "10_tung_anh.csv", encoding="utf-8-sig")
    t0 = ta[(ta.seed == 0) & ~ta.is_bg].copy()
    d = {}
    for key in ["yolo", "tsvm"]:
        g = t0[t0.model == key].set_index("image")
        d[key] = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
    diff = (d["tsvm"] - d["yolo"]).sort_values()
    pick = [diff.index[0], diff.index[len(diff) // 2], diff.index[-1], d["yolo"].sort_values().index[0]]
    pick = list(dict.fromkeys(pick))
    z = {k: np.load(BANG / "du_lieu_dinh_tinh" / f"pred_seed0_{k}.npz") for k in ["yolo", "tsvm"]}
    fig, axes = plt.subplots(len(pick), 4, figsize=(9, 2.3 * len(pick)))
    for r, name in enumerate(pick):
        img = cv2.cvtColor(cv2.imread(str(DATA / "images" / "test" / name)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        gt = np.zeros((h, w), np.uint8)
        for line in (DATA / "labels" / "test" / f"{Path(name).stem}.txt").read_text().splitlines():
            p = np.array(line.split()[1:], np.float32).reshape(-1, 2) * [w, h]
            cv2.fillPoly(gt, [p.astype(np.int32)], 1)
        panels = [("Ảnh", None), ("Nhãn", gt.astype(bool))]
        for k in ["yolo", "tsvm"]:
            m = np.unpackbits(z[k][name])[: h * w].reshape(h, w).astype(bool)
            panels.append((f"{LBL[k]}\nDice {d[k][name]:.3f}", m))
        for c, (title, m) in enumerate(panels):
            ax = axes[r, c]
            show = img.copy()
            if m is not None:
                show[m] = (0.5 * show[m] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)
            ax.imshow(show)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
    fig.suptitle("Minh họa seed 0 trên tập test (vàng: vùng polyp)", fontsize=10)
    fig.tight_layout()
    save(fig, "h05_minh_hoa_seed0")


if __name__ == "__main__":
    main()
