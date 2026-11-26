"""Hình chỉ số huấn luyện và test của 10 seed (YOLO26s-seg và TSVM) cho hướng 2 và hướng 3.

Chỉ đọc results.csv, test_metrics.json (do notebook Kaggle ghi) và bảng pixel đã tính (bang/11_tung_seed.csv).
Không suy luận, không dùng GPU. Ghi <hướng>/ket_qua/danh_gia/hinh/k*.png (300 dpi) và .pdf:

  k01_val_mask_theo_epoch   Precision/Recall/mAP50/mAP50-95 của mask trên val theo epoch (TB, vùng min-max 10 seed)
  k02_val_box_theo_epoch    như trên cho box
  k03_loss_theo_epoch       loss train và val (box, seg, cls) theo epoch
  k04_test_mask_tung_seed   P/R/mAP50/mAP50-95 mask trên test, chấm từng seed nối cặp, kèm Δ, p
  k05_cot_test_ultralytics  biểu đồ cột 8 chỉ số Ultralytics (mask, box) trên test, mean ± SD và chấm từng seed
  k06_cot_pixel             biểu đồ cột Dice/IoU/Precision/Recall theo pixel (H3: test và 3 bộ ngoài)
  k07_chi_phi_tung_seed     giờ huấn luyện và epoch tốt nhất (fitness Ultralytics) theo seed

    python ve_hinh_chi_so_10seed.py            # cả hai hướng
    python ve_hinh_chi_so_10seed.py --huong 3
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

GOC = Path(__file__).resolve().parent
SEEDS = list(range(10))
KEYS = ["yolo", "tsvm"]
C = {"yolo": "#0072B2", "tsvm": "#D55E00"}
LBL = {"yolo": "YOLO26s-seg", "tsvm": "YOLO26s-seg + TSVM"}
HUONG = {
    2: {"ten": "Hướng 2 (BG20 chia 3 tập)", "kq": GOC / "Huong2_Chia3Tap" / "ket_qua",
        "run": {"yolo": lambda s: GOC / "Huong2_Chia3Tap/ket_qua/YOLO26s_seg" / f"H2_BG20_3tap_YOLO26s_seg_s{s}",
                "tsvm": lambda s: GOC / "Huong2_Chia3Tap/ket_qua/TSVM_FP32" / f"H2_BG20_3tap_TSVM_FP32_s{s}"},
        "val": "val 160 ảnh", "test": "test 160 ảnh"},
    3: {"ten": "Hướng 3 (dữ liệu gộp chia 3 tập)", "kq": GOC / "Huong3_GopDuLieu" / "ket_qua",
        "run": {"yolo": lambda s: GOC / "Huong3_GopDuLieu/ket_qua" / f"H3_Gop_3tap_YOLO26s_seg_s{s}",
                "tsvm": lambda s: GOC / "Huong3_GopDuLieu/ket_qua" / f"H3_Gop_3tap_TSVM_FP32_s{s}"},
        "val": "val 532 ảnh", "test": "test 522 ảnh"},
}
CS_MASK = [("precision(M)", "Precision (mask)"), ("recall(M)", "Recall (mask)"), ("mAP50(M)", "mAP50 (mask)"),
           ("mAP50-95(M)", "mAP50–95 (mask)")]
CS_BOX = [("precision(B)", "Precision (box)"), ("recall(B)", "Recall (box)"), ("mAP50(B)", "mAP50 (box)"),
          ("mAP50-95(B)", "mAP50–95 (box)")]
LOSS = [("train/box_loss", "Train box loss"), ("train/seg_loss", "Train seg loss"), ("train/cls_loss", "Train cls loss"),
        ("val/box_loss", "Val box loss"), ("val/seg_loss", "Val seg loss"), ("val/cls_loss", "Val cls loss")]
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
PHAY = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))


def vn(x, n=3):
    return f"{x:.{n}f}".replace(".", ",").replace("-", "−")


def save(fig, hinh, name):
    hinh.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        if ax.axison:
            ax.yaxis.set_major_formatter(PHAY)
    for ext in ("png", "pdf"):
        fig.savefig(hinh / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ", hinh / f"{name}.png")


def ghep_cap(a, b):
    """a = TSVM, b = YOLO (mảng theo seed)."""
    d = a - b
    ok = d.std() > 0
    return {"delta": d.mean(), "p_t": stats.ttest_rel(a, b).pvalue if ok else np.nan,
            "p_w": stats.wilcoxon(a, b).pvalue if ok and (d != 0).any() else np.nan, "hon": int((d > 0).sum())}


def ghi_chu(r):
    return f"Δ = {r['delta']:+.4f}, p t = {vn(r['p_t'], 2)}, TSVM cao hơn {r['hon']}/10".replace(".", ",")


def doc_results(cfg):
    out = {}
    for k in KEYS:
        for s in SEEDS:
            r = pd.read_csv(cfg["run"][k](s) / "results.csv")
            r.columns = [c.strip() for c in r.columns]
            assert len(r) == 100, (k, s, len(r))
            out[(k, s)] = r
    return out


def doc_test(cfg):
    rows = []
    for k in KEYS:
        for s in SEEDS:
            m = json.loads((cfg["run"][k](s) / "test_metrics.json").read_text(encoding="utf-8"))
            rows.append({"model": k, "seed": s, **{c: m[f"metrics/{c}"] for c, _ in CS_MASK + CS_BOX}})
    return pd.DataFrame(rows)


def ve_duong_cong(res, cols, nrow, ncol, tieu_de, hinh, name, ylab_prefix=""):
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 2.8 * nrow), squeeze=False)
    for ax, (col, lab) in zip(axes.flat, cols):
        for k in KEYS:
            a = np.vstack([res[(k, s)][col if "/" in col else f"metrics/{col}"].to_numpy() for s in SEEDS])
            ep = np.arange(1, a.shape[1] + 1)
            ax.plot(ep, a.mean(0), color=C[k], lw=1.2, label=LBL[k])
            ax.fill_between(ep, a.min(0), a.max(0), color=C[k], alpha=0.15)
        ax.set_title(f"{ylab_prefix}{lab}", fontsize=9)
        ax.set_xlabel("Epoch")
        if "loss" in col:  # 2 epoch đầu có loss val rất lớn (seg loss tới ~38): cắt trục theo epoch ≥ 3, ghi chú
            v = np.concatenate([res[(k, s)][col].to_numpy()[2:] for k in KEYS for s in SEEDS])
            lo, hi = v.min(), v.max()
            ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.08 * (hi - lo))
            ax.text(0.98, 0.97, "trục cắt theo epoch ≥ 3", transform=ax.transAxes, ha="right", va="top",
                    fontsize=6.5, color="#777777")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.suptitle(tieu_de, fontsize=10)
    fig.tight_layout()
    save(fig, hinh, name)


def ve_cap_seed(ax, y, t, title):
    for s in y.index:
        ax.plot([0, 1], [y[s], t[s]], color="#999999", lw=0.8, zorder=1)
    for x, k, v in [(0, "yolo", y), (1, "tsvm", t)]:
        ax.scatter(np.full(len(v), x), v, color=C[k], s=20, zorder=2)
        ax.errorbar(x + (-0.12 if x == 0 else 0.12), v.mean(), yerr=v.std(ddof=1), fmt="o", color=C[k], capsize=4,
                    ms=6, zorder=3)
        ax.text(x + (-0.18 if x == 0 else 0.18), v.mean(), f"{vn(v.mean())}\n±{vn(v.std(ddof=1))}",
                ha="right" if x == 0 else "left", va="center", fontsize=7.5, color=C[k])
    ax.set_xticks([0, 1], ["YOLO26s-seg", "+ TSVM"])
    ax.set_xlim(-0.8, 1.8)
    ax.set_title(f"{title}\n{ghi_chu(ghep_cap(t.to_numpy(), y.to_numpy()))}", fontsize=8)


def ve_cot(ax, df, cols, ylim=None):
    """df: model, seed, các cột chỉ số. Cột = mean, thanh = SD, chấm = từng seed."""
    w = 0.38
    for j, k in enumerate(KEYS):
        g = df[df.model == k]
        for i, (c, _) in enumerate(cols):
            v = g[c].to_numpy()
            x = i + (j - 0.5) * w
            ax.bar(x, v.mean(), w * 0.95, yerr=v.std(ddof=1), color=C[k], alpha=0.85, capsize=2.5,
                   label=LBL[k] if i == 0 else None, error_kw={"lw": 0.8})
            ax.scatter(x + np.linspace(-w * 0.25, w * 0.25, len(v)), v, s=5, color="black", alpha=0.55, zorder=3)
            ax.text(x, max(v.mean() + v.std(ddof=1), v.max()) + 0.008, vn(v.mean()), ha="center", fontsize=6.5,
                    rotation=90, va="bottom")
    ax.set_xticks(range(len(cols)), [lab.replace(" (", "\n(") for _, lab in cols], fontsize=8)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", color="#dddddd", lw=0.6)
    ax.set_axisbelow(True)


def mot_huong(h):
    cfg = HUONG[h]
    hinh = cfg["kq"] / "danh_gia" / "hinh"
    ten = cfg["ten"]
    print(ten)
    res = doc_results(cfg)
    test = doc_test(cfg)

    ve_duong_cong(res, CS_MASK, 2, 2, f"{ten}: chỉ số mask trên {cfg['val']} theo epoch (TB, vùng min–max qua 10 seed)",
                  hinh, "k01_val_mask_theo_epoch")
    ve_duong_cong(res, CS_BOX, 2, 2, f"{ten}: chỉ số box trên {cfg['val']} theo epoch (TB, vùng min–max qua 10 seed)",
                  hinh, "k02_val_box_theo_epoch")
    ve_duong_cong(res, LOSS, 2, 3, f"{ten}: loss theo epoch (TB, vùng min–max qua 10 seed)", hinh, "k03_loss_theo_epoch")

    fig, axes = plt.subplots(1, 4, figsize=(14, 3.9))
    for ax, (c, lab) in zip(axes, CS_MASK):
        ve_cap_seed(ax, test[test.model == "yolo"].set_index("seed")[c], test[test.model == "tsvm"].set_index("seed")[c],
                    lab)
    fig.suptitle(f"{ten}: chỉ số mask trên {cfg['test']} (best.pt), 10 seed nối cặp, mean ± SD", fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, hinh, "k04_test_mask_tung_seed")

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ve_cot(ax, test, CS_MASK + CS_BOX, ylim=(0.5, 1.02))
    ax.axvline(3.5, color="#999999", lw=0.8, ls="--")
    ax.legend(frameon=False, loc="upper right", fontsize=8)
    ax.set_ylabel("Giá trị trên tập test")
    ax.set_title(f"{ten}: chỉ số Ultralytics trên {cfg['test']} (mean ± SD, chấm đen = từng seed)", fontsize=10)
    save(fig, hinh, "k05_cot_test_ultralytics")

    sd = pd.read_csv(cfg["kq"] / "danh_gia" / "bang" / "11_tung_seed.csv", encoding="utf-8-sig")
    px = [("dice", "Dice"), ("iou", "IoU"), ("precision", "Precision"), ("recall", "Recall")]
    taps = sorted(sd.tap.unique(), key=["test", "clinicdb", "colondb", "etis"].index) if "tap" in sd else [None]
    ten_tap = {None: cfg["test"], "test": "Test gộp (448 ảnh polyp)", "clinicdb": "CVC-ClinicDB (612)",
               "colondb": "CVC-ColonDB (380)", "etis": "ETIS (196)"}
    fig, axes = plt.subplots(1, len(taps), figsize=(4.6 * len(taps), 3.9), squeeze=False)
    for ax, tp in zip(axes.flat, taps):
        g = sd if tp is None else sd[sd.tap == tp]
        ve_cot(ax, g, px, ylim=(0.5, 1.0))
        r = ghep_cap(g[g.model == "tsvm"].set_index("seed").dice.to_numpy(), g[g.model == "yolo"].set_index("seed").dice.to_numpy())
        ax.set_title(f"{ten_tap[tp]}\nDice: {ghi_chu(r)}", fontsize=8)
    axes.flat[0].set_ylabel("Giá trị theo pixel")
    axes.flat[0].legend(frameon=False, loc="upper left", fontsize=7)
    fig.suptitle(f"{ten}: chỉ số theo pixel (mean ± SD qua 10 seed, chấm đen = từng seed)", fontsize=10, y=1.03)
    fig.tight_layout()
    save(fig, hinh, "k06_cot_pixel")

    # k07: giờ huấn luyện (cột time của results.csv, lượt chạy tiếp có time đặt lại) và epoch tốt nhất theo fitness
    rows = []
    for (k, s), r in res.items():
        tg = r["time"].to_numpy()
        gio = (tg[:-1][tg[1:] < tg[:-1]].sum() + tg[-1]) / 3600
        fit = 0.1 * (r["metrics/mAP50(B)"] + r["metrics/mAP50(M)"]) + 0.9 * (r["metrics/mAP50-95(B)"] + r["metrics/mAP50-95(M)"])
        rows.append({"model": k, "seed": s, "gio": gio, "epoch_tot": int(r.loc[fit.idxmax(), "epoch"])})
    cp = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    for ax, (c, lab) in zip(axes, [("gio", "Giờ huấn luyện mỗi seed (Kaggle T4)"),
                                   ("epoch_tot", "Epoch tốt nhất trên val (fitness Ultralytics)")]):
        for j, k in enumerate(KEYS):
            g = cp[cp.model == k].sort_values("seed")
            ax.bar(g.seed + (j - 0.5) * 0.38, g[c], 0.36, color=C[k], label=f"{LBL[k]}: TB {vn(g[c].mean(), 2 if c == 'gio' else 1)}")
        ax.set_xticks(SEEDS)
        ax.set_xlabel("Seed")
        ax.set_title(lab, fontsize=9)
        ax.set_ylim(0, cp[c].max() * 1.25)
        ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2)
    fig.suptitle(f"{ten}: chi phí và epoch tốt nhất theo seed", fontsize=10)
    fig.tight_layout()
    save(fig, hinh, "k07_chi_phi_tung_seed")
    print(cp.groupby("model")[["gio", "epoch_tot"]].agg(["mean", "min", "max"]).round(2).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--huong", nargs="*", type=int, default=[2, 3], choices=[2, 3])
    a = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    for h in a.huong:
        mot_huong(h)


if __name__ == "__main__":
    main()
