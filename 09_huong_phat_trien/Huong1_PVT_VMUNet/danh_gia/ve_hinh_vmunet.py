"""Hình cho VM-UNet 640 (seed 0–4). Phong cách lấy từ ve_hinh_pvt.py.

Hai nhóm hình, KHÔNG ghi đè hình cũ của 5 mô hình:
  A. Riêng VM-UNet  -> ket_qua/VMUNet_640/hinh/
     v01_duong_cong_vmunet     Val Dice từng epoch của 5 seed + lịch tốc độ học; loss huấn luyện
     v02_val_dice_tot_nhat     Val Dice tốt nhất và epoch đạt được của từng seed
  B. Gộp 6 mô hình  -> ket_qua/so_sanh_6_mo_hinh/hinh/
     s01_cot_pixel_6_mo_hinh   Dice/IoU/Precision/Recall theo pixel (Bảng 4.8)
     s02_dice_tung_seed        Dice từng seed, nối cùng seed
     s03_minh_hoa_seed0        ảnh minh họa seed 0 (cùng 3 ảnh đã chọn cho hình h03 cũ)
     s04_duong_cong_val        Val Dice của notebook theo epoch (PraNet, U-Net, Polyp-PVT, VM-UNet)
     s05_chi_phi_6_mo_hinh     tham số, GFLOPs, giờ huấn luyện
     s06_dice_va_chi_phi       Dice theo GFLOPs, cỡ chấm theo số tham số

    C:\\venv_kl_gpu\\Scripts\\python.exe ve_hinh_vmunet.py [--chi-huan-luyen]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.dont_write_bytecode = True
import ve_hinh_pvt as v                                           # noqa: E402

KQ = v.GOC / "ket_qua"
PIX = KQ / "danh_gia_pixel"
VM = KQ / "VMUNet_640"
OUT_A = VM / "hinh"
OUT_B = KQ / "so_sanh_6_mo_hinh" / "hinh"
OTHER = v.TEST_MAU / "05_so_sanh_mo_hinh_khac"
C_VM = "#E69F00"
MODELS6 = v.MODELS[:4] + [("vmunet640", C_VM, "VM-UNet (VMamba-S)")] + v.MODELS[4:]
NGAN = {"baseline": "YOLO26s-seg", "tsvm": "YOLO + TSVM", "pranet640": "PraNet", "polyppvt640": "Polyp-PVT",
        "vmunet640": "VM-UNet", "unet640": "U-Net"}
LOG = {"pranet640": lambda s: OTHER / "PraNet_640" / f"PraNet_640_aug_s{s}" / "results.csv",
       "unet640": lambda s: OTHER / "UNet_640" / f"UNet_640_aug_s{s}" / "results.csv",
       "polyppvt640": lambda s: KQ / "PolypPVT_640" / f"PolypPVT_640_aug_s{s}" / "results.csv",
       "vmunet640": lambda s: VM / f"VMUNet_640_aug_s{s}" / "results.csv"}
# Chi phí VM-UNet: tham số và FLOPs đếm trên CPU (FlopCounterMode, phép quét không đếm -> cận dưới).
# Độ trễ và VRAM suy luận không đo được công bằng trên laptop vì thiếu kernel CUDA của mamba_ssm.
VM_PARAMS, VM_GFLOPS = 27_427_561, 66.10


def save(fig, out, name):
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    print("  ", out / f"{name}.png")


def doc_log(key):
    return {s: pd.read_csv(LOG[key](s)) for s in range(5)}


def chu_thich(ax, text, y=1.02):
    ax.text(1.0, y, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=v.FONT - 3, color="#555555")


# ================================================================ A. riêng VM-UNet
def v01_duong_cong(logs):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(v.W, v.W * 0.4))
    mau = plt.cm.Oranges(np.linspace(0.45, 0.95, 5))
    for s, r in logs.items():
        a1.plot(r.epoch, r.val_dice, color=mau[s], linewidth=0.9, label=f"seed {s}")
        b = r.val_dice.idxmax()
        a1.scatter(r.epoch[b], r.val_dice[b], s=14, color=mau[s], edgecolors="black", linewidths=0.5, zorder=3)
        a2.plot(r.epoch, r.train_loss, color=mau[s], linewidth=0.9)
    lr = logs[0][["epoch", "lr"]]
    for ax in (a1, a2):
        t = ax.twinx()
        t.plot(lr.epoch, lr.lr * 1e3, color="#777777", linewidth=0.8, linestyle="--")
        t.set_ylim(0, 3.2)
        t.spines["top"].set_visible(False)
        t.tick_params(axis="y", labelsize=v.FONT - 3, colors="#777777")
        if ax is a2:
            t.set_ylabel("tốc độ học (×10⁻³, nét đứt)", fontsize=v.FONT - 3, color="#777777")
        else:
            t.set_yticks([])
        ax.set_xlabel("Epoch")
        v.grid_y(ax)
    a1.set_ylabel("Val Dice (notebook)")
    a1.set_ylim(0, 1)
    v.num_axis(a1, 1)
    a2.set_ylabel("Loss huấn luyện (BCE + Dice)")
    v.num_axis(a2, 1)
    a1.set_title("Val Dice theo epoch", fontsize=v.FONT - 1)
    a2.set_title("Loss huấn luyện", fontsize=v.FONT - 1)
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, l, ncol=5, fontsize=v.FONT - 2, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
    fig.text(1.0, -0.02, "Chấm viền đen: checkpoint tốt nhất. Lịch Cosine T_max = 50 của bản gốc (300 epoch): tốc độ học "
             "giảm đến epoch 50 rồi tăng lại. Batch 4 × cộng dồn 2, ảnh 640, T4.",
             ha="right", va="top", fontsize=v.FONT - 4, color="#555555")
    fig.tight_layout(w_pad=2.5)
    save(fig, OUT_A, "v01_duong_cong_vmunet")


def v02_tot_nhat(logs):
    fig, ax = plt.subplots(figsize=(v.W * 0.7, v.W * 0.38))
    for s, r in logs.items():
        b = r.val_dice.idxmax()
        ax.bar(s, r.val_dice[b], 0.6, color=C_VM, alpha=0.88, zorder=2)
        ax.text(s, 0.51, f"{v.num(r.val_dice[b], 3)}\nep. {int(r.epoch[b])}", ha="center",
                va="bottom", fontsize=v.FONT - 3, color="black")
    m = np.mean([r.val_dice.max() for r in logs.values()])
    sd = np.std([r.val_dice.max() for r in logs.values()], ddof=1)
    ax.axhline(m, color="black", linewidth=0.8, linestyle="--", zorder=3)
    ax.text(-0.45, m + 0.006, f"TB {v.num(m, 3)} ± {v.num(sd, 3)}", ha="left", va="bottom", fontsize=v.FONT - 3)
    ax.set_xlim(-0.5, 4.5)
    ax.set_xticks(range(5), [f"seed {s}" for s in range(5)])
    ax.set_ylim(0.5, 0.92)
    ax.set_ylabel("Val Dice tốt nhất")
    v.num_axis(ax, 2)
    v.grid_y(ax)
    chu_thich(ax, "Val Dice do notebook tính trên 160 ảnh val (ảnh nền đúng khi không báo pixel nào); trục từ 0,5")
    save(fig, OUT_A, "v02_val_dice_tot_nhat")


# ================================================================ B. gộp 6 mô hình
def s01_cot_pixel(sd):
    metrics = [("dice", "Dice"), ("iou", "IoU"), ("precision", "Precision"), ("recall", "Recall")]
    fig, ax = plt.subplots(figsize=(v.W, v.W * 0.55))
    x = np.arange(len(metrics))
    n, w = len(MODELS6), 0.135
    for k, (key, c, lab) in enumerate(MODELS6):
        g = sd[sd.model == key]
        mu = [g[m].mean() for m, _ in metrics]
        s_ = [g[m].std(ddof=1) for m, _ in metrics]
        xs = x + (k - (n - 1) / 2) * w
        ax.bar(xs, mu, w * 0.92, color=c, alpha=0.88, label=lab, zorder=2,
               edgecolor="black" if key in ("polyppvt640", "vmunet640") else "none", linewidth=0.6)
        ax.errorbar(xs, mu, yerr=s_, fmt="none", ecolor="black", elinewidth=0.7, capsize=1.4, zorder=3)
    ax.set_xticks(x, [lab for _, lab in metrics])
    ax.set_ylabel("Giá trị theo pixel")
    ax.set_ylim(0.4, 1.0)
    v.num_axis(ax, 1)
    v.grid_y(ax)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, handlelength=1.2, columnspacing=1.0)
    ax.text(1.0, -0.13, "mean ± SD, seed 0–4; 120 ảnh polyp của tập val; trục tung bắt đầu từ 0,4",
            transform=ax.transAxes, ha="right", va="top", fontsize=v.FONT - 3, color="#555555")
    save(fig, OUT_B, "s01_cot_pixel_6_mo_hinh")


def s02_dice_tung_seed(sd):
    keys = MODELS6
    fig, ax = plt.subplots(figsize=(v.W, v.W * 0.5))
    piv = sd.pivot(index="seed", columns="model", values="dice")
    xs = np.arange(len(keys))
    for s in piv.index:
        ax.plot(xs, [piv.loc[s, k] for k, _, _ in keys], color="#BBBBBB", linewidth=0.7, zorder=1)
    for i, (key, c, lab) in enumerate(keys):
        g = piv[key]
        ax.scatter(np.full(len(g), xs[i]), g, s=22, color=c, edgecolors="white", linewidths=0.4, zorder=3)
        ax.hlines(g.mean(), xs[i] - 0.22, xs[i] + 0.22, color="black", linewidth=1.4, zorder=4)
        ax.text(xs[i] + 0.26, g.mean(), v.num(g.mean(), 3), va="center", fontsize=v.FONT - 2)
    ax.set_xticks(xs, [v.nhan_xuong_dong(lab) for _, _, lab in keys], fontsize=v.FONT - 3)
    ax.set_xlim(-0.4, len(keys) - 0.3)
    ax.set_ylabel("Dice theo pixel (trung bình 120 ảnh)")
    v.num_axis(ax, 2)
    v.grid_y(ax)
    chu_thich(ax, "đường xám nối cùng một seed; vạch đen là trung bình 5 seed")
    save(fig, OUT_B, "s02_dice_tung_seed")


def s03_minh_hoa():
    cu = pd.read_csv(v.PIX_CU / "10_pixel_tung_anh.csv", encoding="utf-8-sig")
    moi = pd.concat([pd.read_csv(PIX / f"10_pixel_tung_anh_{k}.csv", encoding="utf-8-sig")
                     for k in ("polyppvt", "vmunet")], ignore_index=True)
    pix = pd.concat([cu, moi], ignore_index=True)
    pol = pix[(pix.seed == 0) & ~pix.is_bg]
    dice = pol.assign(d=2 * pol.tp / (2 * pol.tp + pol.fp + pol.fn)).pivot(index="image", columns="model", values="d")
    # Giữ đúng 3 ảnh của h03 cũ (chọn theo Dice trung bình của 4 mô hình, không có VM-UNet) để so được hai hình
    k4 = ["baseline", "tsvm", "pranet640", "polyppvt640"]
    tb = dice[k4].mean(axis=1).sort_values()
    picks = [tb.index[-1], tb.index[len(tb) // 2], tb.index[0]]
    keys = [m for m in MODELS6 if m[0] != "unet640"]
    npz = {k: np.load((PIX if k in ("polyppvt640", "vmunet640") else v.PIX_CU) / "du_lieu_dinh_tinh" / f"pred_seed0_{k}.npz")
           for k, _, _ in keys}
    cols = ["Ảnh gốc", "Nhãn"] + [NGAN[k].replace("YOLO + ", "") for k, _, _ in keys]
    fig, axes = plt.subplots(len(picks), len(cols), figsize=(v.W, v.W * 0.45))
    for r, name in enumerate(picks):
        img = cv2.cvtColor(cv2.imread(str(v.DATA / "images" / "val" / name)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        gt = np.zeros((h, w), np.uint8)
        for line in (v.DATA / "labels" / "val" / f"{Path(name).stem}.txt").read_text().splitlines():
            c = np.array(line.split()[1:], np.float32).reshape(-1, 2) * [w, h]
            cv2.fillPoly(gt, [c.astype(np.int32)], 1)
        panels = [img, gt.astype(bool)] + [np.unpackbits(npz[k][name])[: h * w].reshape(h, w).astype(bool)
                                           for k, _, _ in keys]
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
                    ax.text(0.03, 0.04, v.num(dice.loc[name, keys[ci - 2][0]], 2), transform=ax.transAxes,
                            fontsize=v.FONT - 4, color="white", bbox=dict(facecolor="black", alpha=0.55, pad=1, linewidth=0))
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[ci], fontsize=v.FONT - 3)
    fig.subplots_adjust(wspace=0.03, hspace=0.05)
    save(fig, OUT_B, "s03_minh_hoa_seed0")
    return picks


def s04_duong_cong_val():
    keys = [("pranet640", "PraNet"), ("unet640", "U-Net"), ("polyppvt640", "Polyp-PVT"), ("vmunet640", "VM-UNet")]
    mau = {k: c for k, c, _ in MODELS6}
    fig, ax = plt.subplots(figsize=(v.W * 0.85, v.W * 0.45))
    for k, lab in keys:
        L = doc_log(k)
        M = pd.concat([L[s].set_index("epoch").val_dice.rename(s) for s in range(5)], axis=1)
        ax.fill_between(M.index, M.min(1), M.max(1), color=mau[k], alpha=0.15, linewidth=0)
        ax.plot(M.index, M.mean(1), color=mau[k], linewidth=1.2, label=lab)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Val Dice (notebook)")
    ax.set_ylim(0, 1)
    v.num_axis(ax, 1)
    v.grid_y(ax)
    ax.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), handlelength=1.2)
    ax.text(1.0, -0.2, "đường: trung bình 5 seed; vùng nhạt: min–max. Cùng cách tính Val Dice trong notebook "
            "(160 ảnh, ảnh nền đúng khi không báo pixel nào). YOLO/TSVM dùng log Ultralytics nên không vẽ chung.",
            transform=ax.transAxes, ha="right", va="top", fontsize=v.FONT - 4, color="#555555", wrap=True)
    save(fig, OUT_B, "s04_duong_cong_val")


def bang_chi_phi(sd):
    cp = pd.read_csv(KQ / "chi_phi" / "21_flops_do_tre_gpu_5model.csv", encoding="utf-8-sig").set_index("model")
    tg = pd.read_csv(KQ / "chi_phi" / "20_thoi_gian_huan_luyen_tung_seed.csv", encoding="utf-8-sig")
    vm = pd.DataFrame([{"model": "vmunet640", "seed": s, "gio_huan_luyen": doc_log("vmunet640")[s].time_s.iloc[-1] / 3600}
                       for s in range(5)])
    tg = pd.concat([tg[["model", "seed", "gio_huan_luyen"]], vm], ignore_index=True)
    cp.loc["vmunet640", ["params", "gflops_flopcounter"]] = [VM_PARAMS, VM_GFLOPS]
    cp["gio_mean"] = tg.groupby("model").gio_huan_luyen.mean()
    cp["gio_sd"] = tg.groupby("model").gio_huan_luyen.std(ddof=1)
    cp["dice_mean"] = sd.groupby("model").dice.mean()
    cp["dice_sd"] = sd.groupby("model").dice.std(ddof=1)
    return cp, tg


def s05_chi_phi(cp):
    o = [k for k, _, _ in MODELS6]
    mau = [c for _, c, _ in MODELS6]
    panels = [("Số tham số (triệu)", cp.params / 1e6, None, 1),
              ("GFLOPs (ảnh 640×640)", cp.gflops_flopcounter, None, 0),
              ("Huấn luyện (giờ/seed)", cp.gio_mean, cp.gio_sd, 1)]
    fig, axes = plt.subplots(1, 3, figsize=(v.W, v.W * 0.36))
    for ax, (tit, val, err, nd) in zip(axes, panels):
        y = val.reindex(o).values
        x = np.arange(len(o))
        for xi, yi, k, c in zip(x, y, o, mau):
            dam = k in ("polyppvt640", "vmunet640")
            ax.bar(xi, yi, 0.72, color=c, alpha=0.88, zorder=2, edgecolor="black" if dam else c, linewidth=0.6 if dam else 0)
        top = y + (err.reindex(o).values if err is not None else 0)
        if err is not None:
            ax.errorbar(x, y, yerr=err.reindex(o).values, fmt="none", ecolor="black", elinewidth=0.7, capsize=1.6, zorder=3)
        hi = top.max() * 1.18
        ax.set_ylim(0, hi)
        for xi, yi, ti, k in zip(x, y, top, o):
            nhan = v.num(yi, nd) + ("*" if k in ("tsvm", "vmunet640") and "FLOPs" in tit else "")
            ax.text(xi, ti + hi * 0.02, nhan, ha="center", va="bottom", fontsize=v.FONT - 3)
        ax.set_title(tit, fontsize=v.FONT - 1)
        ax.set_xticks(x, [NGAN[k] for k in o], rotation=40, ha="right", fontsize=v.FONT - 3)
        ax.tick_params(axis="y", labelsize=v.FONT - 3)
        v.num_axis(ax, 0)
        v.grid_y(ax)
    fig.text(1.0, -0.03, "* cận dưới: phép quét Mamba không được đếm. Huấn luyện: log Kaggle, mean ± SD seed 0–4; "
             "VM-UNet batch 4 × cộng dồn 2, các mô hình khác batch 8. Độ trễ của VM-UNet chưa đo (không có kernel "
             "mamba_ssm trên Windows).", ha="right", va="top", fontsize=v.FONT - 4, color="#555555", wrap=True)
    fig.tight_layout(w_pad=0.8)
    save(fig, OUT_B, "s05_chi_phi_6_mo_hinh")


def s06_dice_va_chi_phi(cp):
    fig, ax = plt.subplots(figsize=(v.W * 0.8, v.W * 0.5))
    for k, c, lab in MODELS6:
        r = cp.loc[k]
        ax.errorbar(r.gflops_flopcounter, r.dice_mean, yerr=r.dice_sd, fmt="none", ecolor=c, elinewidth=0.8, capsize=2, zorder=2)
        ax.scatter(r.gflops_flopcounter, r.dice_mean, s=25 + r.params / 1e6 * 9, color=c, alpha=0.85, zorder=3,
                   edgecolors="black" if k in ("polyppvt640", "vmunet640") else "white", linewidths=0.6)
        dx, dy, ha = 5, 0.0, "left"
        if k == "tsvm":
            dy = -0.03
        if k == "baseline":
            dy = 0.025
        if k == "pranet640":
            dx, dy = 5, -0.03
        if k == "polyppvt640":
            dx, dy = 4, 0.025
        ax.annotate(f"{NGAN[k]}\n{v.num(r.params / 1e6, 1)} M tham số", (r.gflops_flopcounter, r.dice_mean),
                    xytext=(r.gflops_flopcounter + dx, r.dice_mean + dy), fontsize=v.FONT - 3, ha=ha, va="center")
    ax.set_xlabel("GFLOPs (ảnh 640×640; TSVM, VM-UNet là cận dưới)")
    ax.set_ylabel("Dice theo pixel")
    ax.set_xlim(20, 185)
    lo = cp.dice_mean.min() - 0.06
    ax.set_ylim(lo, 0.95)
    v.num_axis(ax, 2)
    v.grid_y(ax)
    chu_thich(ax, "cỡ chấm theo số tham số; thanh dọc: SD qua seed 0–4")
    save(fig, OUT_B, "s06_dice_va_chi_phi")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi-huan-luyen", action="store_true", help="chỉ vẽ hình không cần kết quả đánh giá pixel")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    v.setup()
    logs = doc_log("vmunet640")
    assert all(len(r) == 100 for r in logs.values())
    v01_duong_cong(logs)
    v02_tot_nhat(logs)
    s04_duong_cong_val()
    if args.chi_huan_luyen:
        return
    sd = pd.read_csv(PIX / "11_pixel_tung_seed_6model.csv", encoding="utf-8-sig")
    assert set(sd.model) == {k for k, _, _ in MODELS6} and sorted(sd.seed.unique()) == [0, 1, 2, 3, 4]
    s01_cot_pixel(sd)
    s02_dice_tung_seed(sd)
    print("   ảnh minh họa:", s03_minh_hoa())
    cp, tg = bang_chi_phi(sd)
    s05_chi_phi(cp)
    s06_dice_va_chi_phi(cp)
    out = KQ / "so_sanh_6_mo_hinh"
    cp.reset_index().to_csv(out / "30_chi_phi_6_mo_hinh.csv", index=False, encoding="utf-8-sig")
    tg.to_csv(out / "31_thoi_gian_huan_luyen_6_mo_hinh.csv", index=False, encoding="utf-8-sig")
    print(cp[["params", "gflops_flopcounter", "gio_mean", "gio_sd", "dice_mean", "dice_sd"]].round(3).to_string())


if __name__ == "__main__":
    main()
