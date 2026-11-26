"""Vẽ hình hướng 3 từ ../ket_qua/danh_gia/bang/ (không suy luận, không GPU).
Ghi ../ket_qua/danh_gia/hinh/<tên>.png (300 dpi) và .pdf. Mẫu theo Huong2_Chia3Tap/danh_gia/ve_hinh_h2.py.

    python ve_hinh_h3.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import cv2

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import danh_gia_h3 as dg  # noqa: E402  (chỉ dùng hằng số và hàm đọc ảnh, không nạp mô hình)

BANG, LUOT, QUAL = dg.OUT, dg.LUOT, dg.QUAL
HINH = dg.KQ / "danh_gia" / "hinh"
C = {"yolo": "#0072B2", "tsvm": "#D55E00"}
LBL = {"yolo": "YOLO26s-seg", "tsvm": "YOLO26s-seg + TSVM"}
TAP = {"test": "Test gộp (448 ảnh polyp)", "clinicdb": "CVC-ClinicDB (612)", "colondb": "CVC-ColonDB (380)",
       "etis": "ETIS (196)"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name):
    HINH.mkdir(parents=True, exist_ok=True)
    from matplotlib.ticker import FuncFormatter
    phay = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ",").replace("-", "−"))
    for ax in fig.axes:  # dấu phẩy thập phân trên trục số
        if ax.axison:
            if not getattr(ax, "truc_y_chu", False):  # trục y là tên nhóm (h03) thì giữ nguyên
                ax.yaxis.set_major_formatter(phay)
            if ax.get_xlabel():
                ax.xaxis.set_major_formatter(phay)
    for ext in ("png", "pdf"):
        fig.savefig(HINH / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  ", HINH / f"{name}.png")


def vn(x, n=3):
    return f"{x:.{n}f}".replace(".", ",")


def paired_ax(ax, sd, metric, title, note):
    y = sd[sd.model == "yolo"].set_index("seed")[metric]
    t = sd[sd.model == "tsvm"].set_index("seed")[metric]
    for s in y.index:
        ax.plot([0, 1], [y[s], t[s]], color="#999999", lw=0.8, zorder=1)
    for x, key, v in [(0, "yolo", y), (1, "tsvm", t)]:
        ax.scatter(np.full(len(v), x), v, color=C[key], s=24, zorder=2)
        ax.errorbar(x + (-0.12 if x == 0 else 0.12), v.mean(), yerr=v.std(ddof=1), fmt="o", color=C[key],
                    capsize=4, ms=6, zorder=3)
        ax.text(x + (-0.18 if x == 0 else 0.18), v.mean(), f"{vn(v.mean())}\n±{vn(v.std(ddof=1))}",
                ha="right" if x == 0 else "left", va="center", fontsize=8, color=C[key])
    ax.set_xticks([0, 1], ["YOLO26s-seg", "+ TSVM"])
    ax.set_xlim(-0.8, 1.8)
    ax.set_title(f"{title}\n{note}", fontsize=9)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sd = pd.read_csv(BANG / "11_tung_seed.csv", encoding="utf-8-sig")
    st = pd.read_csv(BANG / "12_thong_ke.csv", encoding="utf-8-sig")
    stn = pd.read_csv(BANG / "15_thong_ke_theo_nguon_test.csv", encoding="utf-8-sig")

    def note(tap, metric):
        r = st[(st.tap == tap) & (st.metric == metric)].iloc[0]
        return f"Δ = {r.delta:+.4f}, p t = {vn(r.paired_t_p, 2)}, TSVM cao hơn {r.a_higher}/10".replace(".", ",")

    # h01: Dice từng seed, 4 tập
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    for ax, tap in zip(axes, TAP):
        paired_ax(ax, sd[sd.tap == tap], "dice", TAP[tap], note(tap, "dice"))
    axes[0].set_ylabel("Dice theo pixel (trung bình trên ảnh polyp)")
    fig.suptitle("Hướng 3: Dice từng seed, nối cặp theo seed (mean ± SD qua 10 seed)", fontsize=11, y=1.04)
    fig.tight_layout()
    save(fig, "h01_dice_tung_seed_4_tap")

    # h02: mask mAP50–95 trên test
    fig, ax = plt.subplots(figsize=(5.2, 4))
    paired_ax(ax, sd[sd.tap == "test"], "map50_95_mask", "Hướng 3, tập test gộp, 10 seed ghép cặp",
              note("test", "map50_95_mask"))
    ax.set_ylabel("mask mAP50–95 (tập test)")
    save(fig, "h02_map_tung_seed_test")

    # h03: Δ Dice (TSVM − YOLO) với CI 95%: 4 tập + theo nguồn trong test
    rows = [(TAP[t], st[(st.tap == t) & (st.metric == "dice")].iloc[0], "chinh" if t == "test" else "ngoai")
            for t in TAP]
    n_nguon = {"KV": 150, "PG": 209, "SW": 64, "SN": 21, "KA": 4}
    rows += [(f"  test: {dg.NGUON[k]} ({n_nguon[k]})", stn[(stn.nguon == k) & (stn.metric == "dice")].iloc[0], "nguon")
             for k in ["KV", "PG", "SW", "SN", "KA"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    mau = {"chinh": "#222222", "ngoai": "#555555", "nguon": "#aaaaaa"}
    for i, (lab, r, kind) in enumerate(rows):
        yy = len(rows) - 1 - i
        ax.errorbar(r.delta, yy, xerr=[[r.delta - r.ci95_low], [r.ci95_high - r.delta]], fmt="s" if kind == "chinh" else "o",
                    color=mau[kind], capsize=3, ms=7 if kind == "chinh" else 5)
        ax.text(0.078, yy, f"{r.delta:+.4f}  p = {vn(r.paired_t_p, 2)}".replace(".", ","), va="center", fontsize=8)
    ax.axvline(0, color="#888888", lw=0.8, ls="--")
    ax.set_yticks(range(len(rows)), [lab for lab, _, _ in rows][::-1], fontsize=8)
    ax.truc_y_chu = True
    ax.set_xlim(-0.05, 0.11)
    ax.set_xlabel("Δ Dice (TSVM − YOLO26s-seg), CI 95% qua 10 seed")
    ax.set_title("Hướng 3: chênh lệch Dice theo tập và theo nguồn ảnh\n(theo nguồn là thăm dò; SN 21 ảnh, KA 4 ảnh quá nhỏ)",
                 fontsize=9)
    save(fig, "h03_delta_dice_theo_tap")

    # h04: đường cong val (mask mAP50–95 theo epoch)
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for key in ["yolo", "tsvm"]:
        cur = []
        for s in range(10):
            r = pd.read_csv(dg.MODELS[key][1](s) / "results.csv")
            r.columns = [c.strip() for c in r.columns]
            v = r["metrics/mAP50-95(M)"].to_numpy()
            assert len(v) == 100, (key, s, len(v))
            cur.append(v)
        a = np.vstack(cur)
        ep = np.arange(1, a.shape[1] + 1)
        ax.plot(ep, a.mean(0), color=C[key], label=LBL[key])
        ax.fill_between(ep, a.min(0), a.max(0), color=C[key], alpha=0.15)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mask mAP50–95 trên val (532 ảnh)")
    ax.set_title("Hướng 3: đường cong val (trung bình, vùng min–max qua 10 seed)", fontsize=10)
    ax.legend(frameon=False)
    save(fig, "h04_duong_cong_val")

    # h05, h06: minh họa seed 0
    def dice_seed0(tap):
        d = {}
        for key in ["yolo", "tsvm"]:
            g = pd.read_csv(LUOT / f"{tap}_{key}_s0.csv", encoding="utf-8-sig")
            g = g[~g.is_bg].set_index("image")
            d[key] = 2 * g.tp / (2 * g.tp + g.fp + g.fn)
        return d

    def doc_anh(tap, name):
        if tap == "test":
            img = dg.doc(dg.DATA / "images" / "test" / name, cv2.IMREAD_COLOR)
            h, w = img.shape[:2]
            gt = np.zeros((h, w), np.uint8)
            for line in (dg.DATA / "labels" / "test" / f"{Path(name).stem}.txt").read_text().splitlines():
                p = np.array(line.split()[1:], np.float32).reshape(-1, 2) * [w, h]
                cv2.fillPoly(gt, [p.astype(np.int32)], 1)
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), gt.astype(bool)
        _, idir, mdir, _ = dg.NGOAI[tap]
        img = dg.doc(idir / name, cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), dg.doc(mdir / name, cv2.IMREAD_GRAYSCALE) > 127

    def ve_luoi(cases, title, name):
        z = {}
        fig, axes = plt.subplots(len(cases), 4, figsize=(9, 2.3 * len(cases)))
        for r, (tap, img_name, cap, d) in enumerate(cases):
            for k in ["yolo", "tsvm"]:
                if (tap, k) not in z:
                    z[(tap, k)] = np.load(QUAL / f"pred_seed0_{tap}_{k}.npz")
            img, gt = doc_anh(tap, img_name)
            h, w = gt.shape
            panels = [(f"{cap}\n{img_name}", None), ("Nhãn", gt)]
            for k in ["yolo", "tsvm"]:
                m = np.unpackbits(z[(tap, k)][img_name])[: h * w].reshape(h, w).astype(bool)
                panels.append((f"{LBL[k]}\nDice {vn(d[k][img_name])}", m))
            for c, (t, m) in enumerate(panels):
                ax = axes[r, c]
                show = img.copy()
                if m is not None:
                    show[m] = (0.5 * show[m] + 0.5 * np.array([255, 255, 0])).astype(np.uint8)
                ax.imshow(show)
                ax.set_title(t, fontsize=7)
                ax.axis("off")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        save(fig, name)

    d = dice_seed0("test")
    diff = (d["tsvm"] - d["yolo"]).sort_values()
    pick = [(diff.index[0], "TSVM kém hơn nhiều nhất"), (diff.index[len(diff) // 2], "Chênh ở mức trung vị"),
            (diff.index[-1], "TSVM tốt hơn nhiều nhất"), (d["yolo"].sort_values().index[0], "YOLO kém nhất")]
    seen, cases = set(), []
    for n, cap in pick:
        if n not in seen:
            seen.add(n)
            cases.append(("test", n, cap, d))
    ve_luoi(cases, "Hướng 3, seed 0, tập test gộp (vàng: vùng polyp)", "h05_minh_hoa_test_seed0")

    cases = []
    for tap in ["clinicdb", "colondb", "etis"]:
        d = dice_seed0(tap)
        diff = (d["tsvm"] - d["yolo"]).sort_values()
        cases += [(tap, diff.index[0], f"{TAP[tap]}: TSVM kém hơn nhiều nhất", d),
                  (tap, diff.index[-1], f"{TAP[tap]}: TSVM tốt hơn nhiều nhất", d)]
    ve_luoi(cases, "Hướng 3, seed 0, 3 bộ ngoài: ca chênh nhiều nhất mỗi chiều (vàng: vùng polyp)",
            "h06_minh_hoa_bo_ngoai_seed0")


if __name__ == "__main__":
    main()
