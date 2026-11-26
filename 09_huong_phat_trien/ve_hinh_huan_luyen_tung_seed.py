"""Hình quá trình HUẤN LUYỆN (train/val) của từng seed, YOLO26s-seg so với TSVM, cho hướng 2 và hướng 3.

Chỉ đọc results.csv do Ultralytics ghi khi huấn luyện trên Kaggle (loss train, loss và chỉ số trên val mỗi epoch).
Không dùng tập test, không dùng bộ ngoài, không dùng GPU. Ghi <hướng>/ket_qua/hinh_huan_luyen/:

  t01..t04  val Precision / Recall / mAP50 / mAP50-95 (mask) theo epoch, mỗi ô một seed (2 x 5 ô)
  t05       loss phân đoạn (seg) train và val theo epoch, mỗi ô một seed
  t06       cột: chỉ số mask trên val tại epoch tốt nhất (epoch mà best.pt được lưu), từng seed
  t07       cột: chỉ số box trên val tại epoch tốt nhất, từng seed
  t08       cột: loss cuối (epoch 100) train và val, từng seed

    python ve_hinh_huan_luyen_tung_seed.py            # cả hai hướng
    python ve_hinh_huan_luyen_tung_seed.py --huong 3
"""
from __future__ import annotations

import argparse
import sys

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ve_hinh_chi_so_10seed import C, CS_BOX, CS_MASK, HUONG, KEYS, LBL, PHAY, SEEDS, doc_results, vn  # noqa: E402

plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, hinh, name):
    hinh.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        if ax.axison:
            ax.yaxis.set_major_formatter(PHAY)
    for ext in ("png", "pdf"):
        fig.savefig(hinh / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ", hinh / f"{name}.png")


def epoch_tot(r):
    """Epoch có fitness Ultralytics lớn nhất (box + mask) = epoch best.pt được lưu."""
    fit = 0.1 * (r["metrics/mAP50(B)"] + r["metrics/mAP50(M)"]) + 0.9 * (r["metrics/mAP50-95(B)"] + r["metrics/mAP50-95(M)"])
    return int(fit.idxmax())


def luoi_seed(res, cot, tieu_de, hinh, name, loss=False):
    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
    for ax, s in zip(axes.flat, SEEDS):
        gt = []
        for k in KEYS:
            r = res[(k, s)]
            ep = r["epoch"].to_numpy()
            for c, ls in cot:
                v = r[c].to_numpy()
                ax.plot(ep, v, color=C[k], lw=0.9, ls=ls)
                gt.append(v[2:])
            if not loss:
                b = epoch_tot(r)
                v = r.loc[b, cot[0][0]]
                ax.scatter(r.loc[b, "epoch"], v, color=C[k], s=18, zorder=3, edgecolor="white", lw=0.5)
        gt = np.concatenate(gt)
        lo, hi = gt.min(), (np.percentile(gt, 99) if loss else gt.max())  # loss: bỏ vài điểm vọt đầu
        ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
        ax.set_title(f"Seed {s}", fontsize=9)
        ax.grid(color="#eeeeee", lw=0.5)
    for ax in axes[1]:
        ax.set_xlabel("Epoch")
    tay = [plt.Line2D([], [], color=C[k], label=LBL[k]) for k in KEYS]
    if loss:
        tay += [plt.Line2D([], [], color="#555555", ls="-", label="train"),
                plt.Line2D([], [], color="#555555", ls="--", label="val")]
        ghi = "trục cắt ở phân vị 99% để thấy rõ phần sau, vài điểm vọt ở các epoch đầu nằm ngoài khung"
    else:
        tay += [plt.Line2D([], [], color="#555555", marker="o", ls="", label="epoch tốt nhất (best.pt)")]
        ghi = "trục cắt theo epoch ≥ 3"
    fig.legend(handles=tay, loc="upper center", ncol=len(tay), frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=8)
    fig.suptitle(f"{tieu_de} ({ghi})", fontsize=10, y=1.04)
    fig.tight_layout()
    save(fig, hinh, name)


def cot_seed(bang, cols, tieu_de, hinh, name):
    """bang: model, seed, cột chỉ số. Mỗi ô một chỉ số, trục x = seed, 2 cột/seed, thêm cột TB."""
    fig, axes = plt.subplots(len(cols), 1, figsize=(11, 2.3 * len(cols)), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, (c, lab) in zip(axes, cols):
        allv = []
        for j, k in enumerate(KEYS):
            g = bang[bang.model == k].sort_values("seed")
            v = g[c].to_numpy()
            allv.append(v)
            x = np.arange(10) + (j - 0.5) * 0.38
            ax.bar(x, v, 0.36, color=C[k], label=LBL[k])
            ax.bar(10.6 + (j - 0.5) * 0.38, v.mean(), 0.36, yerr=v.std(ddof=1), color=C[k], capsize=3, alpha=0.85,
                   hatch="//", edgecolor="white")
            ax.text(10.6 + (j - 0.5) * 0.38, v.mean() + v.std(ddof=1), vn(v.mean()), ha="center", va="bottom",
                    fontsize=6.5, rotation=90)
        a, b = allv[1], allv[0]
        d = a - b
        p = stats.ttest_rel(a, b).pvalue if d.std() > 0 else np.nan
        lo, hi = min(x.min() for x in allv), max(x.max() for x in allv)
        tren = hi + 0.6 * (hi - lo) + 1e-3
        ax.set_ylim(lo - 0.6 * (hi - lo) - 1e-3, tren if "loss" in c else min(tren, 1.0))
        ax.set_ylabel(lab, fontsize=8)
        ax.text(0.01, 0.95, f"TSVM − YOLO: Δ = {d.mean():+.4f}, p t = {vn(p, 2)}, TSVM cao hơn {(d > 0).sum()}/10".replace(".", ","),
                transform=ax.transAxes, va="top", fontsize=7.5)
        ax.grid(axis="y", color="#eeeeee", lw=0.5)
        ax.set_axisbelow(True)
    axes[-1].set_xticks(list(range(10)) + [10.6], [f"Seed {s}" for s in SEEDS] + ["TB ± SD"])
    fig.tight_layout()
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(tieu_de, fontsize=10, y=1.05)
    save(fig, hinh, name)


def ve_o_tb(ax, res, col, lab):
    """Một ô: TB 10 seed (đường) và min-max (vùng mờ) của YOLO và TSVM."""
    gt = []
    for k in KEYS:
        a = np.vstack([res[(k, s)][col].to_numpy() for s in SEEDS])
        ep = np.arange(1, a.shape[1] + 1)
        ax.fill_between(ep, a.min(0), a.max(0), color=C[k], alpha=0.18, lw=0)
        ax.plot(ep, a.mean(0), color=C[k], lw=1.1, label=f"{LBL[k]} (TB 10 seed)")
        gt.append(a[:, 2:].ravel())
    gt = np.concatenate(gt)
    lo, hi = ((gt.min(), np.percentile(gt, 99.5)) if "loss" in col
              else (np.percentile(gt, 1), gt.max()))  # bỏ vài điểm cực trị ở các epoch đầu
    ax.set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    ax.set_title(lab, fontsize=9)
    ax.grid(color="#eeeeee", lw=0.5)


def tong_hop_tb(res, ten, cfg, hinh):
    """r00: một hình kiểu results.png, mỗi ô = TB 10 seed (đường) và min-max (vùng mờ) của YOLO và TSVM chồng lên nhau."""
    r0 = res[("yolo", 0)]
    tr = [c for c in r0.columns if c.startswith("train/") and any((res[(k, s)][c] != 0).any() for k in KEYS for s in SEEDS)]
    va = [c.replace("train/", "val/") for c in tr if c.replace("train/", "val/") in r0.columns
          and any((res[(k, s)][c.replace("train/", "val/")] != 0).any() for k in KEYS for s in SEEDS)]  # val sem_loss luôn 0
    ten_loss = {"box_loss": "box loss", "seg_loss": "seg loss", "cls_loss": "cls loss", "l1_loss": "L1 loss",
                "sem_loss": "sem loss", "dfl_loss": "DFL loss"}
    hang = [[(c, f"Train {ten_loss.get(c.split('/')[1], c)}") for c in tr],
            [(c, f"Val {ten_loss.get(c.split('/')[1], c)}") for c in va],
            [(f"metrics/{c}", f"Val {lab}") for c, lab in CS_MASK],
            [(f"metrics/{c}", f"Val {lab}") for c, lab in CS_BOX]]
    ncol = max(len(x) for x in hang)
    stt = [0]
    fig, axes = plt.subplots(4, ncol, figsize=(3.3 * ncol, 10.5))
    for i, cols in enumerate(hang):
        for j in range(ncol):
            ax = axes[i, j]
            if j >= len(cols):
                ax.axis("off")
                continue
            col, lab = cols[j]
            ve_o_tb(ax, res, col, lab)
            if i == 3 or (i == 1 and j >= len(hang[2])):
                ax.set_xlabel("Epoch")
            # Bản riêng: mỗi biểu đồ một hình (thư mục r_tung_bieu_do/)
            f1, a1 = plt.subplots(figsize=(6.2, 4))
            ve_o_tb(a1, res, col, lab)
            a1.set_title(f"{ten}\n{lab}: trung bình 10 seed (vùng mờ: min–max)", fontsize=9.5)
            a1.set_xlabel("Epoch")
            a1.set_ylabel(lab)
            a1.legend(frameon=False, fontsize=8, loc="upper right" if "loss" in col else "lower right")
            stt[0] += 1
            save(f1, hinh / "r_tung_bieu_do", f"r{stt[0]:02d}_{lab.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('–', '_')}")
    h_, l_ = axes[0, 0].get_legend_handles_labels()
    h_ += [plt.Rectangle((0, 0), 1, 1, color="#999999", alpha=0.3)]
    l_ += ["vùng mờ: min–max qua 10 seed"]
    fig.tight_layout()
    fig.legend(h_, l_, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{ten}: kết quả huấn luyện trung bình 10 seed (train và {cfg['val']}). "
                 "Trục cắt bỏ vài điểm cực trị ở các epoch đầu để thấy rõ phần sau", fontsize=11, y=1.06)
    save(fig, hinh, "r00_ket_qua_huan_luyen_tb_10seed")


def mot_huong(h):
    cfg = HUONG[h]
    hinh = cfg["kq"] / "hinh_huan_luyen"
    ten = cfg["ten"]
    print(ten)
    res = doc_results(cfg)
    tong_hop_tb(res, ten, cfg, hinh)

    for i, (c, lab) in enumerate(CS_MASK, 1):
        luoi_seed(res, [(f"metrics/{c}", "-")], f"{ten}: {lab} trên {cfg['val']} theo epoch, từng seed", hinh,
                  f"t0{i}_val_{c.replace('(M)', '_mask').replace('-', '_')}_tung_seed")
    luoi_seed(res, [("train/seg_loss", "-"), ("val/seg_loss", "--")], f"{ten}: seg loss train và val theo epoch, từng seed",
              hinh, "t05_seg_loss_tung_seed", loss=True)

    rows = []
    for (k, s), r in res.items():
        b = epoch_tot(r)
        last = r.iloc[-1]
        rows.append({"model": k, "seed": s, "epoch_tot": int(r.loc[b, "epoch"]),
                     **{c: r.loc[b, f"metrics/{c}"] for c, _ in CS_MASK + CS_BOX},
                     **{f"cuoi_{c}": last[c] for c in ["train/box_loss", "train/seg_loss", "train/cls_loss",
                                                       "val/box_loss", "val/seg_loss", "val/cls_loss"]}})
    bang = pd.DataFrame(rows)
    hinh.mkdir(parents=True, exist_ok=True)
    bang.round(5).to_csv(hinh / "val_tai_epoch_tot_nhat.csv", index=False, encoding="utf-8-sig")
    cot_seed(bang, [(c, lab) for c, lab in CS_MASK],
             f"{ten}: chỉ số mask trên {cfg['val']} tại epoch tốt nhất (best.pt), từng seed", hinh, "t06_cot_val_mask_tung_seed")
    cot_seed(bang, [(c, lab) for c, lab in CS_BOX],
             f"{ten}: chỉ số box trên {cfg['val']} tại epoch tốt nhất (best.pt), từng seed", hinh, "t07_cot_val_box_tung_seed")
    cot_seed(bang, [("cuoi_train/seg_loss", "Train seg loss"), ("cuoi_val/seg_loss", "Val seg loss"),
                    ("cuoi_train/box_loss", "Train box loss"), ("cuoi_val/box_loss", "Val box loss")],
             f"{ten}: loss ở epoch cuối (100), từng seed", hinh, "t08_cot_loss_cuoi_tung_seed")
    with pd.option_context("display.width", 200):
        print(bang.groupby("model")[[c for c, _ in CS_MASK] + ["epoch_tot"]].agg(["mean", "std"]).round(4).to_string())


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
