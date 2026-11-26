"""Một biểu đồ: val mask mAP50-95 theo epoch (trung bình qua các seed) của YOLO26s-seg và TSVM
cho 3 thiết kế: báo cáo khóa luận (2 tập, 20 seed), hướng 2 (BG20 chia 3 tập, 10 seed), hướng 3 (dữ liệu gộp, 10 seed).

Chỉ đọc results.csv (Test_Mau chỉ đọc). Ghi so_sanh_bao_cao_H2_H3/ (PNG 300 dpi, PDF, CSV tóm tắt).
Lưu ý: ba thiết kế dùng ba tập val khác nhau, nên mức tuyệt đối không so trực tiếp được; chỉ so hình dạng đường cong
và khoảng cách YOLO - TSVM trong cùng một thiết kế.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from ve_hinh_chi_so_10seed import HUONG, PHAY, vn  # noqa: E402

GOC = Path(__file__).resolve().parent
TM = GOC.parent / "Test_Mau" / "03_ket_qua_huan_luyen"
OUT = GOC / "so_sanh_bao_cao_H2_H3"
COL = "metrics/mAP50-95(M)"
THIET_KE = {
    "bc": ("Báo cáo khóa luận (train/val, val 160 ảnh, 20 seed)", "#7F7F7F", range(20),
           {"yolo": lambda s: TM / "Baseline_YOLO26s_seg" / f"Kvasir_BG20_Baseline_YOLO26s_seg_s{s}_w2",
            "tsvm": lambda s: TM / "TSVM_FP32" / f"Kvasir_BG20_YOLO26s_seg_TSVM_FP32_s{s}_w2"}),
    "h2": ("Hướng 2 (BG20 chia 3 tập, val 160 ảnh, 10 seed)", "#0072B2", range(10), HUONG[2]["run"]),
    "h3": ("Hướng 3 (dữ liệu gộp chia 3 tập, val 532 ảnh, 10 seed)", "#009E73", range(10), HUONG[3]["run"]),
}
KIEU = {"yolo": ("-", "YOLO26s-seg"), "tsvm": ("--", "YOLO26s-seg + TSVM")}


def doc(f):
    r = pd.read_csv(f / "results.csv")
    r.columns = [c.strip() for c in r.columns]
    assert len(r) == 100, (f, len(r))
    return r


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5.6))
    rows = []
    for key, (ten, mau, seeds, run) in THIET_KE.items():
        tot = {}
        for m, (ls, lab) in KIEU.items():
            rs = [doc(run[m](s)) for s in seeds]
            a = np.vstack([r[COL].to_numpy() for r in rs])
            ep = np.arange(1, 101)
            ax.fill_between(ep, a.min(0), a.max(0), color=mau, alpha=0.07, lw=0)
            ax.plot(ep, a.mean(0), color=mau, ls=ls, lw=1.4, label=f"{ten.split(' (')[0]}: {lab}")
            fit = [0.1 * (r["metrics/mAP50(B)"] + r["metrics/mAP50(M)"]) + 0.9 * (r["metrics/mAP50-95(B)"] + r[COL]) for r in rs]
            tot[m] = np.array([r.loc[f.idxmax(), COL] for r, f in zip(rs, fit)])
            rows.append({"thiet_ke": ten, "model": lab, "n_seed": len(seeds),
                         "val_map50_95_epoch_tot_TB": tot[m].mean(), "SD": tot[m].std(ddof=1),
                         "val_map50_95_epoch100_TB": a[:, -1].mean(), "epoch_dat_95pct_dinh_TB":
                             float(np.mean([np.argmax(x >= 0.95 * x.max()) + 1 for x in a]))})
        d = tot["tsvm"] - tot["yolo"]
        rows[-1].update(delta_tsvm_yolo=d.mean(), p_t=stats.ttest_rel(tot["tsvm"], tot["yolo"]).pvalue,
                        tsvm_cao_hon=f"{(d > 0).sum()}/{len(d)}")
    ax.set_ylim(0.45, 0.85)
    ax.set_xlim(0, 101)
    ax.yaxis.set_major_formatter(PHAY)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val mAP50–95 (mask)")
    ax.grid(color="#eeeeee", lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=1, loc="lower right")
    ax.set_title("Val mAP50–95 (mask) theo epoch: báo cáo khóa luận, hướng 2, hướng 3\n"
                 "đường = trung bình các seed, vùng mờ = min–max; nét liền YOLO26s-seg, nét đứt + TSVM\n"
                 "Ba thiết kế dùng ba tập val khác nhau: chỉ so dạng đường cong và khoảng cách trong cùng thiết kế",
                 fontsize=9.5)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"so_sanh_map50_95_bao_cao_h2_h3.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    t = pd.DataFrame(rows)
    t.round(4).to_csv(OUT / "tom_tat_map50_95_val.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 250, "display.max_columns", 20):
        print(t.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
