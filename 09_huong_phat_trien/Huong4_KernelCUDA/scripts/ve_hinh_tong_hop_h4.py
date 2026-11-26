"""Hướng 4: hình tổng hợp.
h4_01a–f (6 tệp)  : YOLO26s-seg, TSVM PyTorch, TSVM kernel CUDA, 20 seed (val mAP, giờ huấn luyện, Dice test, Dice dữ liệu ngoài,
                     báo nhầm ảnh không polyp, Δ CUDA − PyTorch với CI 95%).
h4_02a–c (3 tệp)  : 6 mô hình của Hướng 1 (seed 0–4) thêm TSVM CUDA (cùng seed 0–4): Dice pixel test, giờ huấn luyện, Dice theo giờ.
Mọi số đọc từ CSV gốc. Ghi ket_qua_moc3/hinh/*.png (300 dpi) và *.pdf.
"""
import glob, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

sys.stdout.reconfigure(encoding="utf-8")
H4 = Path(__file__).resolve().parents[1]
TM = H4.parents[1] / "Test_Mau"
H1 = H4.parent / "Huong1_PVT_VMUNet" / "ket_qua"
KQ = H4 / "ket_qua_moc3"
OUT = KQ / "hinh"; OUT.mkdir(exist_ok=True)
P = TM / "00_tai_lieu_tong_quan" / "bang" / "pixel_640"
S = range(20)
K = "metrics/mAP50-95(M)"
MAU = {"baseline": "#8c8c8c", "tsvm": "#4C72B0", "cuda": "#DD8452"}
TEN = {"baseline": "YOLO26s-seg", "tsvm": "TSVM PyTorch", "cuda": "TSVM kernel CUDA"}
phay = lambda nd: FuncFormatter(lambda v, _: f"{v:.{nd}f}".replace(".", ","))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def csv_run(m, s):
    if m == "baseline":
        return TM / "03_ket_qua_huan_luyen" / "Baseline_YOLO26s_seg" / f"Kvasir_BG20_Baseline_YOLO26s_seg_s{s}_w2" / "results.csv"
    if m == "tsvm":
        return TM / "03_ket_qua_huan_luyen" / "TSVM_FP32" / f"Kvasir_BG20_YOLO26s_seg_TSVM_FP32_s{s}_w2" / "results.csv"
    return Path(glob.glob(str(KQ / "tai_ve_kaggle" / "*" / f"Kvasir_BG20_YOLO26s_seg_TSVM_CUDA_s{s}_w2" / "results.csv"))[0])


def mo(m, s):
    d = pd.read_csv(csv_run(m, s)); d.columns = [c.strip() for c in d.columns]
    gio = float(open(csv_run(m, s).parent / "gio.txt").read()) if m == "cuda" else d["time"].iloc[-1] / 3600
    return d[K].max(), gio


def tach(hang, cot, co):
    """Mỗi ô là một hình riêng (một biểu đồ một tệp)."""
    figs = np.empty((hang, cot), object); ax = np.empty((hang, cot), object)
    for i in range(hang):
        for j in range(cot):
            figs[i, j], ax[i, j] = plt.subplots(figsize=co)
    return figs, ax


def luu(figs, ax, ten):
    import re
    for (i, j), f in np.ndenumerate(figs):
        a = ax[i, j]
        a.set_title(re.sub(r"^\([a-z]\) ", "", a.get_title(loc="left")), fontweight="bold", loc="left")
        f.tight_layout()
        for e in ("png", "pdf"):
            f.savefig(OUT / f"{ten[i][j]}.{e}", dpi=300)
        plt.close(f)


# ---------------- số liệu 3 mô hình, 20 seed
V = {m: np.array([mo(m, s) for s in S]) for m in MAU}               # [:,0] mAP, [:,1] giờ
pt = pd.read_csv(P / "11_pixel_tung_seed.csv", encoding="utf-8-sig")
pc = pd.read_csv(KQ / "pixel_test" / "11_pixel_tung_seed.csv", encoding="utf-8-sig")
TEST = {m: pt[pt.model == m].set_index("seed").loc[S] for m in ("baseline", "tsvm")}
TEST["cuda"] = pc.set_index("seed").loc[S]
en = pd.read_csv(P / "du_lieu_ngoai" / "20seed" / "01_ngoai_tung_seed.csv", encoding="utf-8-sig")
ec = pd.read_csv(KQ / "du_lieu_ngoai" / "01_cuda_tung_seed.csv", encoding="utf-8-sig")
BO = [("clinicdb", "CVC-ClinicDB"), ("colondb", "CVC-ColonDB"), ("etis", "ETIS"), ("polypgen", "PolypGen")]
EXT = {m: {k: en[(en.model == m) & (en.dataset == k)].set_index("seed").loc[S] for k, _ in BO} for m in ("baseline", "tsvm")}
EXT["cuda"] = {k: ec[ec.dataset == k].set_index("seed").loc[S] for k, _ in BO}
cb = pd.read_csv(P / "nen_cecum" / "06_nen_cecum_tung_seed.csv", encoding="utf-8-sig")
cc = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in (KQ / "du_lieu_ngoai" / "nen_cecum_tung_luot").glob("cecum800_tsvm_s*.csv")])
CEC = {m: cb[cb.model == m].set_index("seed").loc[S, "fp_image_rate"].to_numpy() for m in ("baseline", "tsvm")}
CEC["cuda"] = cc.groupby("seed").pred_px.apply(lambda x: (x > 0).mean()).loc[S].to_numpy()

figs, ax = tach(2, 3, (6.2, 4.4))
ms = list(MAU)


def strip(a, data, nd, ylab, title):
    rng = np.random.default_rng(0)
    for i, m in enumerate(ms):
        y = data[m]
        a.scatter(i + rng.uniform(-0.12, 0.12, len(y)), y, s=14, color=MAU[m], alpha=0.75)
        a.errorbar(i + 0.28, y.mean(), yerr=y.std(ddof=1), fmt="D", color="k", capsize=3, ms=4)
        a.text(i + 0.36, y.mean(), f"{y.mean():.{nd}f}".replace(".", ","), va="center", fontsize=8)
    a.set_xticks(range(3), [TEN[m].replace(" ", "\n", 1) for m in ms]); a.set_xlim(-0.5, 2.85)
    a.yaxis.set_major_formatter(phay(nd - 1)); a.set_ylabel(ylab); a.set_title(title, fontweight="bold", loc="left")


strip(ax[0, 0], {m: V[m][:, 0] for m in ms}, 4, "mask mAP50-95 tốt nhất", "(a) Val trong huấn luyện, 20 seed")
g = [V[m][:, 1] for m in ms]
ax[0, 1].bar(range(3), [x.mean() for x in g], yerr=[x.std(ddof=1) for x in g], color=[MAU[m] for m in ms], capsize=4)
for i, x in enumerate(g):
    ax[0, 1].text(i, x.mean() + 0.2, f"{x.mean():.2f} giờ".replace(".", ","), ha="center")
ax[0, 1].set_xticks(range(3), [TEN[m].replace(" ", "\n", 1) for m in ms]); ax[0, 1].set_ylabel("Giờ huấn luyện / seed (100 epoch)")
ax[0, 1].yaxis.set_major_formatter(phay(0)); ax[0, 1].set_title("(b) Thời gian huấn luyện, 20 seed", fontweight="bold", loc="left")
strip(ax[0, 2], {m: TEST[m].dice.to_numpy() for m in ms}, 4, "Dice theo pixel (120 ảnh polyp)", "(c) Tập test Kvasir-SEG, 20 seed")

a = ax[1, 0]; w = 0.26
for i, m in enumerate(ms):
    y = [EXT[m][k].dice.mean() for k, _ in BO]; e = [EXT[m][k].dice.std(ddof=1) for k, _ in BO]
    a.bar(np.arange(4) + (i - 1) * w, y, w, yerr=e, color=MAU[m], capsize=2, label=TEN[m])
a.set_xticks(range(4), [t for _, t in BO]); a.set_ylim(0.4, 0.85); a.yaxis.set_major_formatter(phay(2))
a.set_ylabel("Dice theo pixel"); a.set_title("(d) Dữ liệu khác nguồn, 20 seed", fontweight="bold", loc="left"); a.legend(frameon=False, fontsize=8)

a = ax[1, 1]
FP = [("Nền test\n(40 ảnh)", {m: TEST[m].bg_fp_image_rate.to_numpy() for m in ms}),
      ("Manh tràng\n(800 ảnh)", CEC),
      ("PolypGen không polyp\n(125 ảnh)", {m: EXT[m]["polypgen"].bg_fp_image_rate.to_numpy() for m in ms})]
for i, m in enumerate(ms):
    a.bar(np.arange(3) + (i - 1) * w, [100 * d[m].mean() for _, d in FP], w, yerr=[100 * d[m].std(ddof=1) for _, d in FP],
          color=MAU[m], capsize=2, label=TEN[m])
a.set_xticks(range(3), [t for t, _ in FP]); a.legend(frameon=False, fontsize=8, loc="upper left"); a.set_ylabel("% ảnh không polyp bị báo nhầm")
a.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))); a.set_title("(e) Báo nhầm trên ảnh không có polyp, 20 seed", fontweight="bold", loc="left")

a = ax[1, 2]
F = [("Val mAP50-95", V["tsvm"][:, 0], V["cuda"][:, 0]), ("Test Dice", TEST["tsvm"].dice.to_numpy(), TEST["cuda"].dice.to_numpy())] + \
    [(f"{t} Dice", EXT["tsvm"][k].dice.to_numpy(), EXT["cuda"][k].dice.to_numpy()) for k, t in BO]
for j, (t, x, y) in enumerate(F[::-1]):
    d = y - x; ci = stats.t.ppf(0.975, 19) * d.std(ddof=1) / np.sqrt(20); p = stats.ttest_rel(y, x).pvalue
    a.errorbar(d.mean(), j, xerr=ci, fmt="o", color=MAU["cuda"], capsize=3)
    a.text(0.0405, j, f"p = {p:.2f}".replace(".", ","), va="center", fontsize=8)
a.axvline(0, color="k", lw=0.8); a.set_yticks(range(len(F)), [t for t, _, _ in F[::-1]]); a.set_xlim(-0.035, 0.05)
a.xaxis.set_major_formatter(phay(2)); a.set_xlabel("Δ = CUDA − PyTorch (trung bình, CI 95%)")
a.set_title("(f) TSVM CUDA − TSVM PyTorch, ghép cặp 20 seed", fontweight="bold", loc="left")
luu(figs, ax, [["h4_01a_val_map_20seed", "h4_01b_gio_huan_luyen", "h4_01c_dice_test_20seed"],
              ["h4_01d_dice_du_lieu_ngoai", "h4_01e_bao_nham_anh_khong_polyp", "h4_01f_chenh_lech_cuda_pytorch"]])

# ---------------- 7 mô hình, seed 0–4 (cùng giao thức Hướng 1)
s6 = pd.read_csv(H1 / "danh_gia_pixel" / "11_pixel_tung_seed_6model.csv", encoding="utf-8-sig")
cp = pd.read_csv(H1 / "so_sanh_6_mo_hinh" / "30_chi_phi_6_mo_hinh.csv", encoding="utf-8-sig").set_index("model")
D7 = {m: s6[s6.model == m].set_index("seed").loc[range(5), "dice"].to_numpy() for m in cp.index}
D7["cuda"] = TEST["cuda"].loc[range(5), "dice"].to_numpy()
G7 = {m: cp.loc[m, "gio_mean"] for m in cp.index}; G7["cuda"] = V["cuda"][:5, 1].mean()
T7 = {"baseline": "YOLO26s-seg", "tsvm": "TSVM PyTorch", "cuda": "TSVM CUDA", "pranet640": "PraNet",
      "unet640": "U-Net", "polyppvt640": "Polyp-PVT", "vmunet640": "VM-UNet"}
M7 = {**MAU, "pranet640": "#55A868", "unet640": "#C44E52", "polyppvt640": "#8172B3", "vmunet640": "#937860"}
od = sorted(D7, key=lambda m: -D7[m].mean())
figs, ax = tach(1, 3, (6.2, 4.4)); ax = ax[0]
a = ax[0]
a.barh(range(7), [D7[m].mean() for m in od], xerr=[D7[m].std(ddof=1) for m in od], color=[M7[m] for m in od], capsize=3)
for i, m in enumerate(od):
    a.text(D7[m].mean() + D7[m].std(ddof=1) + 0.004, i, f"{D7[m].mean():.3f}".replace(".", ","), va="center", fontsize=8)
a.set_yticks(range(7), [T7[m] for m in od]); a.invert_yaxis(); a.set_xlim(0.6, 0.95); a.xaxis.set_major_formatter(phay(2))
a.set_xlabel("Dice theo pixel, tập test (seed 0–4)"); a.set_title("(a) Bảy mô hình: Dice trên tập test", fontweight="bold", loc="left")
og = sorted(G7, key=lambda m: G7[m])
a = ax[1]
a.barh(range(7), [G7[m] for m in og], color=[M7[m] for m in og])
for i, m in enumerate(og):
    a.text(G7[m] + 0.2, i, f"{G7[m]:.1f}".replace(".", ","), va="center", fontsize=8)
a.set_yticks(range(7), [T7[m] for m in og]); a.invert_yaxis(); a.xaxis.set_major_formatter(phay(0))
a.set_xlabel("Giờ huấn luyện / seed"); a.set_title("(b) Bảy mô hình: giờ huấn luyện mỗi seed", fontweight="bold", loc="left")
a = ax[2]
for m in D7:
    a.errorbar(G7[m], D7[m].mean(), yerr=D7[m].std(ddof=1), fmt="o", color=M7[m], ms=8, capsize=3)
    a.annotate(T7[m], (G7[m], D7[m].mean()), xytext={"baseline": (8, 8), "cuda": (8, -14), "tsvm": (6, -12)}.get(m, (6, 4)), textcoords="offset points", fontsize=8)
a.annotate("", xy=(G7["cuda"], D7["cuda"].mean()), xytext=(G7["tsvm"], D7["tsvm"].mean()),
           arrowprops=dict(arrowstyle="->", color=MAU["cuda"], lw=1.2))
a.set_xlabel("Giờ huấn luyện / seed"); a.set_ylabel("Dice theo pixel"); a.xaxis.set_major_formatter(phay(0)); a.yaxis.set_major_formatter(phay(2))
a.set_title("(c) Bảy mô hình: Dice theo giờ huấn luyện", fontweight="bold", loc="left")
luu(figs, ax.reshape(1, 3), [["h4_02a_dice_7_mo_hinh", "h4_02b_gio_7_mo_hinh", "h4_02c_dice_theo_gio_7_mo_hinh"]])
print({T7[m]: (round(D7[m].mean(), 4), round(G7[m], 2)) for m in od})
print("Đã ghi", OUT)
