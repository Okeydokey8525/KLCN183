"""Hướng 4, mốc 3: so sánh ghép cặp 20 seed TSVM bản PyTorch (báo cáo) với bản kernel CUDA.

Endpoint chính: mask mAP50-95 tốt nhất trên val (như báo cáo). Phụ (thăm dò): mAP50, box mAP50-95,
trung bình mask mAP50-95 10 epoch cuối, giờ huấn luyện.
Ghi ket_qua_moc3/{bang_20seed.csv, thong_ke.json, hinh_*.png/pdf}.
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8")
H4 = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TM = os.path.join(H4, "..", "..", "Test_Mau", "03_ket_qua_huan_luyen", "TSVM_FP32")
OUT = os.path.join(H4, "ket_qua_moc3")
K = "metrics/mAP50-95(M)"


def doc(csv):
    d = pd.read_csv(csv); d.columns = [c.strip() for c in d.columns]
    i = d[K].idxmax()
    return {"map": d.loc[i, K], "map50": d.loc[i, "metrics/mAP50(M)"], "box": d.loc[i, "metrics/mAP50-95(B)"],
            "epoch": int(d.loc[i, "epoch"]), "tb10": d[K].iloc[-10:].mean()}


hang = []
for s in range(20):
    pt = doc(os.path.join(TM, f"Kvasir_BG20_YOLO26s_seg_TSVM_FP32_s{s}_w2", "results.csv"))
    pt["gio"] = pd.read_csv(os.path.join(TM, f"Kvasir_BG20_YOLO26s_seg_TSVM_FP32_s{s}_w2", "results.csv"))["time"].iloc[-1] / 3600
    run = glob.glob(os.path.join(OUT, "tai_ve_kaggle", "*", f"Kvasir_BG20_YOLO26s_seg_TSVM_CUDA_s{s}_w2"))
    assert len(run) == 1, (s, run)
    cu = doc(os.path.join(run[0], "results.csv"))
    cu["gio"] = float(open(os.path.join(run[0], "gio.txt")).read())
    hang.append({"seed": s, **{f"{k}_pt": v for k, v in pt.items()}, **{f"{k}_cuda": v for k, v in cu.items()}})
B = pd.DataFrame(hang)
B.round(5).to_csv(os.path.join(OUT, "bang_20seed.csv"), index=False)


def ghep(a, b):
    d = b - a; n = len(d); se = d.std(ddof=1) / np.sqrt(n)
    ci = stats.t.ppf(0.975, n - 1) * se
    return {"pt_mean": a.mean(), "pt_sd": a.std(ddof=1), "cuda_mean": b.mean(), "cuda_sd": b.std(ddof=1),
            "delta": d.mean(), "ci95": [d.mean() - ci, d.mean() + ci], "p_t": stats.ttest_rel(b, a).pvalue,
            "p_wilcoxon": stats.wilcoxon(b, a).pvalue, "dz": d.mean() / d.std(ddof=1),
            "cuda_thang": int((d > 0).sum()), "n": n,
            "p_levene": stats.levene(a, b, center="median").pvalue}


TK = {m: ghep(B[f"{m}_pt"], B[f"{m}_cuda"]) for m in ("map", "map50", "box", "tb10", "gio")}
TK["epoch_tot_nhat"] = {"pt_mean": B.epoch_pt.mean(), "cuda_mean": B.epoch_cuda.mean()}
json.dump(TK, open(os.path.join(OUT, "thong_ke.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

TEN = {"map": "mask mAP50-95 tốt nhất (CHÍNH)", "map50": "mask mAP50", "box": "box mAP50-95", "tb10": "mask mAP50-95 TB 10 epoch cuối", "gio": "giờ huấn luyện"}
for m, t in TK.items():
    if m in TEN:
        print(f"{TEN[m]:34s} PT {t['pt_mean']:.4f}±{t['pt_sd']:.4f} | CUDA {t['cuda_mean']:.4f}±{t['cuda_sd']:.4f} | "
              f"Δ {t['delta']:+.4f} CI [{t['ci95'][0]:+.4f},{t['ci95'][1]:+.4f}] | p_t {t['p_t']:.3g} p_W {t['p_wilcoxon']:.3g} "
              f"dz {t['dz']:+.2f} | CUDA thắng {t['cuda_thang']}/20 | Levene {t['p_levene']:.3g}")
print("epoch tốt nhất TB: PT", TK["epoch_tot_nhat"]["pt_mean"], "CUDA", TK["epoch_tot_nhat"]["cuda_mean"])
print(B[["seed", "map_pt", "map_cuda", "epoch_pt", "epoch_cuda", "gio_pt", "gio_cuda"]].round(4).to_string(index=False))

# Hình
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.4, 1]})
for _, r in B.iterrows():
    ax[0].plot([0, 1], [r.map_pt, r.map_cuda], color="#bbbbbb", lw=0.8, zorder=1)
ax[0].scatter(np.zeros(20), B.map_pt, color="#4C72B0", zorder=2, label="PyTorch (báo cáo)")
ax[0].scatter(np.ones(20), B.map_cuda, color="#DD8452", zorder=2, label="Kernel CUDA")
for x, c in ((0, "map_pt"), (1, "map_cuda")):
    ax[0].errorbar(x + (-0.12 if x == 0 else 0.12), B[c].mean(), yerr=B[c].std(ddof=1), fmt="D", color="k", capsize=4)
ax[0].set_xticks([0, 1], ["PyTorch", "Kernel CUDA"]); ax[0].set_xlim(-0.4, 1.4)
ax[0].set_ylabel("mask mAP50-95 tốt nhất (val)"); ax[0].set_title("20 seed, đường xám nối cùng seed")
t = TK["map"]
ax[0].text(0.5, 0.02, f"Δ = {t['delta']:+.4f}  (CI95 {t['ci95'][0]:+.4f}…{t['ci95'][1]:+.4f}),  p_t = {t['p_t']:.2f}".replace(".", ","),
           transform=ax[0].transAxes, ha="center", fontsize=9)
ax[0].set_ylim(B[["map_pt","map_cuda"]].min().min()-0.006, None)
from matplotlib.ticker import FuncFormatter
ax[0].yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:.3f}".replace(".", ",")))
ax[1].yaxis.set_major_formatter(FuncFormatter(lambda v,_: f"{v:g}".replace(".", ",")))
g = TK["gio"]
ax[1].bar([0, 1], [g["pt_mean"], g["cuda_mean"]], yerr=[g["pt_sd"], g["cuda_sd"]], color=["#4C72B0", "#DD8452"], capsize=5)
for x, v in ((0, g["pt_mean"]), (1, g["cuda_mean"])):
    ax[1].text(x, v + 0.15, f"{v:.2f} giờ".replace(".", ","), ha="center")
ax[1].set_xticks([0, 1], ["PyTorch", "Kernel CUDA"]); ax[1].set_ylabel("Giờ huấn luyện / seed")
ax[1].set_title("Thời gian huấn luyện (100 epoch)")
fig.tight_layout()
for e in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"hinh_moc3_20seed.{e}"), dpi=300)
print("Đã ghi", OUT)
