"""Đánh giá theo pixel cho Polyp-PVT 640 (seed 0–4) trên 160 ảnh validation BG20, cùng giao thức Bảng 4.8.

Giao thức chép từ Test_Mau/00_tai_lieu_tong_quan/scripts/danh_gia_pixel.py (không import file đó):
- Nhãn chuẩn chính: mặt nạ vẽ từ polygon YOLO ở độ phân giải gốc (cv2.fillPoly). Nhãn phụ: mặt nạ gốc Kvasir-SEG.
- Tiền xử lý giống notebook huấn luyện: kéo thẳng về 640×640, chuẩn hóa ImageNet.
- Đầu ra: logit P1 + P2 (đúng hàm xac_suat của notebook), phóng song tuyến về kích thước gốc, sigmoid > 0,5.
- Chỉ số tính trên từng ảnh của 120 ảnh polyp; 40 ảnh nền tính tỉ lệ ảnh có ít nhất một pixel bị báo là polyp.

Chỉ ĐỌC từ Test_Mau (dữ liệu BG20, mặt nạ Kvasir-SEG, bảng 10_pixel_tung_anh.csv của 4 mô hình cũ).
Mọi kết quả GHI vào Baseline_PVT_VMUNet/ket_qua/danh_gia_pixel/.

Chạy (cần GPU và gói timm trong venv):
    C:\\venv_kl_gpu\\Scripts\\python.exe danh_gia_pixel_pvt.py                 # suy luận 5 seed + tổng hợp + so sánh
    C:\\venv_kl_gpu\\Scripts\\python.exe danh_gia_pixel_pvt.py --seeds 0 --tag _thu   # chạy thử 1 seed
    C:\\venv_kl_gpu\\Scripts\\python.exe danh_gia_pixel_pvt.py --tu-bang      # chỉ tổng hợp lại từ bảng đã có
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

HERE = Path(__file__).resolve().parent
GOC = HERE.parent                                            # Baseline_PVT_VMUNet/Huong1_PVT_VMUNet
TEST_MAU = GOC.parent.parent / "Test_Mau"
DATA = TEST_MAU / "01_du_lieu" / "Kvasir_YOLO_SEG_BG20"
RAW_MASKS = TEST_MAU / "01_du_lieu" / "tao_BG20" / "Kvasir-SEG" / "masks"
BANG_CU = TEST_MAU / "00_tai_lieu_tong_quan" / "bang" / "pixel_640" / "10_pixel_tung_anh.csv"
PVT_REPO = GOC / "ma_nguon_goc" / "Polyp-PVT-main"
PVT_W = GOC.parent / "trong_so_tai_ve" / "pvt_v2_b2.pth"
CKPT = lambda s: GOC / "ket_qua" / "PolypPVT_640" / f"PolypPVT_640_aug_s{s}" / f"polyp_pvt_best_s{s}.pth"
OUT = GOC / "ket_qua" / "danh_gia_pixel"

SEEDS = [0, 1, 2, 3, 4]
SIZE = 640
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
NHAN = {"polyppvt640": "Polyp-PVT 640", "baseline": "YOLO26s-seg", "tsvm": "YOLO26s-seg + TSVM",
        "pranet640": "PraNet (ResNet-50) 640", "unet640": "U-Net 640"}
SO_VOI = ["tsvm", "baseline", "pranet640"]                   # so ghép cặp Polyp-PVT với từng mô hình này
METRICS = ["dice", "iou", "precision", "recall", "dice_micro", "iou_micro", "bg_fp_image_rate",
           "raw_dice", "raw_iou", "raw_precision", "raw_recall"]


# ------------------------------------------------------------------ dữ liệu (giống danh_gia_pixel.py)
def load_val():
    items = []
    for p in sorted((DATA / "images" / "val").glob("*.jpg")):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gt = np.zeros((h, w), np.uint8)
        lbl = DATA / "labels" / "val" / f"{p.stem}.txt"
        if lbl.stat().st_size > 0:
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) > 5:
                    c = np.array(parts[1:], np.float32).reshape(-1, 2)
                    c[:, 0] *= w
                    c[:, 1] *= h
                    cv2.fillPoly(gt, [c.astype(np.int32)], 1)
        raw = None
        if not p.stem.startswith("bg_"):
            m = cv2.imread(str(RAW_MASKS / f"{p.stem}.jpg"), cv2.IMREAD_GRAYSCALE)
            raw = (m > 127).astype(np.uint8)
            assert raw.shape == gt.shape, p
        items.append({"name": p.name, "bgr": img, "gt": gt.astype(bool),
                      "raw": None if raw is None else raw.astype(bool), "is_bg": p.stem.startswith("bg_")})
    assert len(items) == 160 and sum(i["is_bg"] for i in items) == 40
    return items


# ------------------------------------------------------------------ mô hình
def tao_polyp_pvt():
    """Dựng PolypPVT bằng mã gốc của tác giả. Hàm khởi tạo gốc nạp cứng './pretrained_pth/pvt_v2_b2.pth':
    chuyển lời gọi đó sang file pvt_v2_b2.pth trong thư mục này. Trọng số thật nạp sau từ checkpoint (strict=True)."""
    if str(PVT_REPO) not in sys.path:
        sys.path.insert(0, str(PVT_REPO))
    from lib.pvt import PolypPVT
    goc = torch.load
    torch.load = lambda *a, **k: goc(PVT_W, map_location="cpu", weights_only=False)
    try:
        return PolypPVT()
    finally:
        torch.load = goc


def du_doan(net, items, dev):
    preds = []
    with torch.no_grad():
        for it in items:
            h, w = it["gt"].shape
            rgb = cv2.cvtColor(it["bgr"], cv2.COLOR_BGR2RGB)
            x = cv2.resize(rgb, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            x = (x - MEAN) / STD
            t = torch.from_numpy(x).permute(2, 0, 1)[None].to(dev)
            p1, p2 = net(t)
            logit = F.interpolate(p1 + p2, size=(h, w), mode="bilinear", align_corners=False)
            preds.append((torch.sigmoid(logit)[0, 0] > 0.5).cpu().numpy())
    return preds


# ------------------------------------------------------------------ chỉ số (giống danh_gia_pixel.py)
def counts(pred, gt):
    return (int(np.logical_and(pred, gt).sum()), int(np.logical_and(pred, ~gt).sum()),
            int(np.logical_and(~pred, gt).sum()))


def per_image_metrics(tp, fp, fn):
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    return dice, iou, prec, rec


def theo_seed(df):
    rows = []
    for (key, s), g in df.groupby(["model", "seed"], sort=False):
        pol, bg = g[~g.is_bg], g[g.is_bg]
        r = {"model": key, "seed": s}
        for pre, (tpc, fpc, fnc) in [("", ("tp", "fp", "fn")), ("raw_", ("raw_tp", "raw_fp", "raw_fn"))]:
            m = np.array([per_image_metrics(a, b, c) for a, b, c in zip(pol[tpc], pol[fpc], pol[fnc])])
            r[f"{pre}dice"], r[f"{pre}iou"], r[f"{pre}precision"], r[f"{pre}recall"] = np.nanmean(m, 0)
            TP, FP, FN = pol[tpc].sum(), pol[fpc].sum(), pol[fnc].sum()
            r[f"{pre}dice_micro"] = 2 * TP / (2 * TP + FP + FN)
            r[f"{pre}iou_micro"] = TP / (TP + FP + FN)
        r["polyp_imgs_no_pred"] = int((pol.pred_px == 0).sum())
        r["bg_fp_image_rate"] = float((bg.pred_px > 0).mean())
        r["bg_fp_area_mean"] = float((bg.pred_px / bg.img_px).mean())
        rows.append(r)
    return pd.DataFrame(rows)


def thong_ke(sd):
    """mean ± SD từng mô hình (seed 0–4) và so ghép cặp Polyp-PVT − mô hình khác."""
    rows = []
    for key in sd.model.unique():
        g = sd[sd.model == key]
        for m in METRICS:
            rows.append({"model": key, "label": NHAN.get(key, key), "metric": m, "n_seeds": len(g),
                         "mean": g[m].mean(), "sd": g[m].std(ddof=1), "min": g[m].min(), "max": g[m].max()})
    p = sd[sd.model == "polyppvt640"].set_index("seed")
    for other in [k for k in SO_VOI if k in set(sd.model)]:
        o = sd[sd.model == other].set_index("seed")
        common = p.index.intersection(o.index)
        if len(common) < 3:
            continue
        for m in METRICS:
            d = p.loc[common, m] - o.loc[common, m]
            ok = d.std() > 0
            rows.append({"model": f"polyppvt640_vs_{other}", "metric": m, "n_seeds": len(common),
                         "mean": d.mean(), "sd": d.std(ddof=1),
                         "paired_t_p": stats.ttest_rel(p.loc[common, m], o.loc[common, m]).pvalue if ok else np.nan,
                         "wilcoxon_p": stats.wilcoxon(p.loc[common, m], o.loc[common, m]).pvalue if ok else np.nan,
                         "pvt_higher": int((d > 0).sum())})
    return pd.DataFrame(rows)


def tong_hop(df_pvt, out, tag):
    out.mkdir(parents=True, exist_ok=True)
    df_pvt.to_csv(out / f"10_pixel_tung_anh_polyppvt{tag}.csv", index=False, encoding="utf-8-sig")
    seeds = sorted(df_pvt.seed.unique())
    cu = pd.read_csv(BANG_CU, encoding="utf-8-sig")
    cu = cu[cu.seed.isin(seeds) & cu.model.isin(["baseline", "tsvm", "pranet640", "unet640"])]
    sd = theo_seed(pd.concat([df_pvt, cu], ignore_index=True))
    sd.to_csv(out / f"11_pixel_tung_seed_5model{tag}.csv", index=False, encoding="utf-8-sig")
    st = thong_ke(sd)
    st.to_csv(out / f"12_pixel_thong_ke_5model{tag}.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 220, "display.max_rows", 300):
        print(st[st.metric.isin(["dice", "iou", "precision", "recall", "bg_fp_image_rate", "raw_dice"])].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=SEEDS)
    ap.add_argument("--tag", default="", help="hậu tố tên file (khi chạy thử)")
    ap.add_argument("--tu-bang", action="store_true", help="không suy luận; tổng hợp lại từ 10_pixel_tung_anh_polyppvt.csv")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.tu_bang:
        df = pd.read_csv(OUT / f"10_pixel_tung_anh_polyppvt{args.tag}.csv", encoding="utf-8-sig")
        tong_hop(df[df.seed.isin(args.seeds)], OUT, args.tag)
        return

    for p in [DATA, RAW_MASKS, BANG_CU, PVT_REPO / "lib" / "pvt.py", PVT_W] + [CKPT(s) for s in args.seeds]:
        assert p.exists(), f"Thiếu: {p}"
    assert torch.cuda.is_available(), "Cần GPU (chạy bằng C:\\venv_kl_gpu\\Scripts\\python.exe)"
    dev = "cuda:0"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    import timm
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| timm", timm.__version__)

    items = load_val()
    print(f"Validation: {len(items)} ảnh ({sum(not i['is_bg'] for i in items)} polyp, {sum(i['is_bg'] for i in items)} nền)")
    qual = OUT / "du_lieu_dinh_tinh"
    qual.mkdir(parents=True, exist_ok=True)

    rows, t0 = [], time.time()
    for s in args.seeds:
        t1 = time.time()
        net = tao_polyp_pvt()
        net.load_state_dict(torch.load(CKPT(s), map_location="cpu", weights_only=True), strict=True)
        net = net.to(dev).eval()
        preds = du_doan(net, items, dev)
        del net
        torch.cuda.empty_cache()
        for it, pr in zip(items, preds):
            tp, fp, fn = counts(pr, it["gt"])
            row = {"model": "polyppvt640", "seed": s, "image": it["name"], "is_bg": it["is_bg"],
                   "pred_px": int(pr.sum()), "gt_px": int(it["gt"].sum()), "img_px": it["gt"].size,
                   "tp": tp, "fp": fp, "fn": fn}
            if it["raw"] is not None:
                rtp, rfp, rfn = counts(pr, it["raw"])
                row.update(raw_tp=rtp, raw_fp=rfp, raw_fn=rfn)
            rows.append(row)
        if s == 0:   # mặt nạ seed 0 cho hình định tính, cùng định dạng pred_seed0_<model>.npz của báo cáo
            np.savez_compressed(qual / "pred_seed0_polyppvt640.npz",
                                **{it["name"]: np.packbits(pr) for it, pr in zip(items, preds)},
                                **{f"__shape__{it['name']}": np.array(pr.shape) for it, pr in zip(items, preds)})
        print(f"  polyppvt640 seed {s}: {time.time() - t1:5.1f}s")
    print(f"Tổng thời gian suy luận: {(time.time() - t0) / 60:.1f} phút")
    tong_hop(pd.DataFrame(rows), OUT, args.tag)


if __name__ == "__main__":
    main()
