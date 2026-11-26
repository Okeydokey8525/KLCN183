"""Hướng 2: đánh giá theo pixel trên TẬP TEST (160 ảnh = val cũ) cho YOLO26s-seg và TSVM, seed 0–9.

Giao thức chép từ Test_Mau/00_tai_lieu_tong_quan/scripts/danh_gia_pixel.py (không import file đó), giống Bảng 4.8:
- YOLO: imgsz 640, conf 0,25, iou 0,7, retina_masks=True, half=False; gộp mọi mặt nạ của ảnh thành một.
- Nhãn chính: polygon YOLO vẽ ở độ phân giải gốc (cv2.fillPoly). Nhãn phụ: mặt nạ gốc Kvasir-SEG (> 127).
- Chỉ số tính trên 120 ảnh polyp; 40 ảnh nền tính tỉ lệ ảnh có ít nhất một pixel bị dự đoán là polyp.
- Mã mô hình: fork Test_Mau/02_ma_nguon/ultralytics_TSVM_FP32 (chỉ đọc, nạp qua junction tạm).

Chỉ đọc Test_Mau và checkpoint; mọi kết quả ghi vào ../ket_qua/danh_gia/bang/. Dừng nếu file đầu ra đã có.

    python danh_gia_test_h2.py --seeds 0 --models yolo --tag _thu     # chạy thử
    python danh_gia_test_h2.py                                         # 20 checkpoint
    python danh_gia_test_h2.py --tu-bang                               # chỉ tổng hợp lại từ 10_*.csv
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
H2 = HERE.parent                                              # Huong2_Chia3Tap
TEST_MAU = H2.parents[1] / "Test_Mau"
DATA = H2 / "du_lieu" / "Kvasir_BG20_3tap"
RAW_MASKS = TEST_MAU / "01_du_lieu" / "tao_BG20" / "Kvasir-SEG" / "masks"
FORK = TEST_MAU / "02_ma_nguon" / "ultralytics_TSVM_FP32"
KQ = H2 / "ket_qua"
OUT = KQ / "danh_gia" / "bang"
OLD = TEST_MAU / "00_tai_lieu_tong_quan" / "bang" / "pixel_640" / "10_pixel_tung_anh.csv"
MODELS = {
    "yolo": ("YOLO26s-seg", lambda s: KQ / "YOLO26s_seg" / f"H2_BG20_3tap_YOLO26s_seg_s{s}"),
    "tsvm": ("YOLO26s-seg + TSVM", lambda s: KQ / "TSVM_FP32" / f"H2_BG20_3tap_TSVM_FP32_s{s}"),
}
OLD_KEY = {"yolo": "baseline", "tsvm": "tsvm"}
METRICS = ["dice", "iou", "precision", "recall", "dice_micro", "iou_micro", "bg_fp_image_rate",
           "raw_dice", "raw_iou", "map50_95_mask", "map50_mask", "map50_95_box"]


def load_test():
    items = []
    for p in sorted((DATA / "images" / "test").glob("*.jpg")):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gt = np.zeros((h, w), np.uint8)
        lbl = DATA / "labels" / "test" / f"{p.stem}.txt"
        if lbl.exists() and lbl.stat().st_size > 0:
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
            raw = (m > 127).astype(bool)
            assert raw.shape == gt.shape, p
        items.append({"name": p.name, "bgr": img, "gt": gt.astype(bool), "raw": raw, "is_bg": p.stem.startswith("bg_")})
    assert len(items) == 160 and sum(i["is_bg"] for i in items) == 40, len(items)
    return items


def import_fork():
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    sys.dont_write_bytecode = True
    tmp = Path(tempfile.mkdtemp(prefix="tsvm_fork_"))
    link = tmp / "ultralytics"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(FORK)], check=True, capture_output=True)
    sys.path.insert(0, str(tmp))
    import ultralytics  # noqa: F401
    assert Path(ultralytics.__file__).resolve().parent == FORK.resolve(), ultralytics.__file__

    def cleanup():
        subprocess.run(["cmd", "/c", "rmdir", str(link)], check=False, capture_output=True)
        os.rmdir(tmp)
    return cleanup


def predict_yolo(weights, items, dev):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    preds = []
    for it in items:
        h, w = it["gt"].shape
        r = model.predict(it["bgr"], imgsz=640, conf=0.25, iou=0.7, retina_masks=True, half=False,
                          device=dev, verbose=False)[0]
        m = np.zeros((h, w), bool)
        if r.masks is not None and len(r.masks):
            mm = r.masks.data.cpu().numpy() > 0.5
            assert mm.shape[1:] == (h, w), (mm.shape, h, w)
            m = mm.any(0)
        preds.append(m)
    del model
    return preds


def counts(pred, gt):
    return (int(np.logical_and(pred, gt).sum()), int(np.logical_and(pred, ~gt).sum()),
            int(np.logical_and(~pred, gt).sum()))


def per_image_metrics(tp, fp, fn):
    dice = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else np.nan
    iou = tp / (tp + fp + fn) if (tp + fp + fn) else np.nan
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else np.nan
    return dice, iou, prec, rec


def per_seed(df: pd.DataFrame, with_map: bool) -> pd.DataFrame:
    rows = []
    for (key, s), g in df.groupby(["model", "seed"], sort=False):
        pol, bg = g[~g.is_bg], g[g.is_bg]
        r = {"model": key, "seed": int(s)}
        for pre, (a, b, c) in [("", ("tp", "fp", "fn")), ("raw_", ("raw_tp", "raw_fp", "raw_fn"))]:
            m = np.array([per_image_metrics(x, y, z) for x, y, z in zip(pol[a], pol[b], pol[c])])
            r[f"{pre}dice"], r[f"{pre}iou"], r[f"{pre}precision"], r[f"{pre}recall"] = np.nanmean(m, 0)
            TP, FP, FN = pol[a].sum(), pol[b].sum(), pol[c].sum()
            r[f"{pre}dice_micro"] = 2 * TP / (2 * TP + FP + FN)
            r[f"{pre}iou_micro"] = TP / (TP + FP + FN)
        r["polyp_imgs_no_pred"] = int((pol.pred_px == 0).sum())
        r["bg_fp_image_rate"] = float((bg.pred_px > 0).mean())
        r["bg_fp_area_mean"] = float((bg.pred_px / bg.img_px).mean())
        if with_map:
            tm = json.loads((MODELS[key][1](s) / "test_metrics.json").read_text(encoding="utf-8"))
            r["map50_95_mask"] = tm["metrics/mAP50-95(M)"]
            r["map50_mask"] = tm["metrics/mAP50(M)"]
            r["map50_95_box"] = tm["metrics/mAP50-95(B)"]
            r["gio_huan_luyen"] = tm.get("gio_huan_luyen")
            res = pd.read_csv(MODELS[key][1](s) / "results.csv")
            res.columns = [c.strip() for c in res.columns]
            fit = 0.1 * res["metrics/mAP50(M)"] + 0.9 * res["metrics/mAP50-95(M)"]
            r["best_epoch_val"] = int(res.loc[fit.idxmax(), "epoch"])
            r["val_map50_95_mask_best"] = float(res.loc[fit.idxmax(), "metrics/mAP50-95(M)"])
        rows.append(r)
    return pd.DataFrame(rows)


def paired(a: pd.Series, b: pd.Series):
    d = a - b
    n = len(d)
    ci = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    ok = d.std() > 0
    return {"n_seeds": n, "mean_a": a.mean(), "sd_a": a.std(ddof=1), "mean_b": b.mean(), "sd_b": b.std(ddof=1),
            "delta": d.mean(), "sd_delta": d.std(ddof=1), "ci95_low": d.mean() - ci, "ci95_high": d.mean() + ci,
            "paired_t_p": stats.ttest_rel(a, b).pvalue if ok else np.nan,
            "wilcoxon_p": stats.wilcoxon(a, b).pvalue if ok else np.nan,
            "cohen_dz": d.mean() / d.std(ddof=1) if ok else np.nan,
            "a_higher": int((d > 0).sum()), "levene_bf_p": stats.levene(a, b, center="median").pvalue}


def tong_hop(df: pd.DataFrame, tag: str):
    sd = per_seed(df, with_map=True)
    sd.to_csv(OUT / f"11_tung_seed{tag}.csv", index=False, encoding="utf-8-sig")
    rows = []
    # (1) mô hình mới: TSVM so với YOLO, ghép theo seed
    if {"yolo", "tsvm"} <= set(sd.model):
        t, y = sd[sd.model == "tsvm"].set_index("seed"), sd[sd.model == "yolo"].set_index("seed")
        common = t.index.intersection(y.index)
        if len(common) >= 3:
            for m in METRICS:
                rows.append({"so_sanh": "H2: TSVM - YOLO", "metric": m, **paired(t.loc[common, m], y.loc[common, m])})
    # (2) mới (tách val) so với cũ (chọn checkpoint trên chính tập này), cùng 160 ảnh, cùng seed 0–9
    if OLD.exists():
        old = pd.read_csv(OLD, encoding="utf-8-sig")
        old = old[old.model.isin(OLD_KEY.values()) & old.seed.between(0, 9)]
        old = old.assign(model=old.model.map({v: k for k, v in OLD_KEY.items()}))
        so = per_seed(old, with_map=False)
        so.to_csv(OUT / f"13_bao_cao_cu_seed0_9{tag}.csv", index=False, encoding="utf-8-sig")
        for key in sd.model.unique():
            n, o = sd[sd.model == key].set_index("seed"), so[so.model == key].set_index("seed")
            common = n.index.intersection(o.index)
            if len(common) >= 3:
                for m in [x for x in METRICS if x in o.columns]:
                    rows.append({"so_sanh": f"{MODELS[key][0]}: H2 - báo cáo cũ", "metric": m,
                                 **paired(n.loc[common, m], o.loc[common, m])})
    st = pd.DataFrame(rows)
    st.to_csv(OUT / f"12_thong_ke{tag}.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 250, "display.max_rows", 200):
        print(sd[["model", "seed", "dice", "iou", "precision", "recall", "bg_fp_image_rate", "map50_95_mask"]].to_string(index=False))
        if len(st):
            print(st[st.metric.isin(["dice", "iou", "bg_fp_image_rate", "map50_95_mask", "raw_dice"])][
                ["so_sanh", "metric", "mean_a", "sd_a", "mean_b", "sd_b", "delta", "ci95_low", "ci95_high",
                 "paired_t_p", "wilcoxon_p", "cohen_dz", "a_higher"]].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    ap.add_argument("--models", nargs="*", default=["yolo", "tsvm"], choices=list(MODELS))
    ap.add_argument("--tag", default="")
    ap.add_argument("--tu-bang", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    f10 = OUT / f"10_tung_anh{args.tag}.csv"
    if args.tu_bang:
        tong_hop(pd.read_csv(f10, encoding="utf-8-sig"), args.tag)
        return
    if f10.exists():
        raise SystemExit(f"Đã có {f10.name}, không ghi đè. Dùng --tag khác hoặc --tu-bang.")
    import torch
    assert torch.cuda.is_available(), "Cần GPU (C:\\venv_kl_gpu\\Scripts\\python.exe)"
    dev = "cuda:0"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    items = load_test()
    print(f"Test: {len(items)} ảnh ({sum(not i['is_bg'] for i in items)} polyp, {sum(i['is_bg'] for i in items)} nền)")
    cleanup = import_fork()
    import ultralytics
    print("ultralytics:", ultralytics.__version__, ultralytics.__file__)
    qual = OUT / "du_lieu_dinh_tinh"
    qual.mkdir(exist_ok=True)
    rows, t0 = [], time.time()
    try:
        for key in args.models:
            for s in args.seeds:
                w = MODELS[key][1](s) / "weights" / "best.pt"
                t1 = time.time()
                preds = predict_yolo(w, items, dev)
                torch.cuda.empty_cache()
                for it, pr in zip(items, preds):
                    tp, fp, fn = counts(pr, it["gt"])
                    row = {"model": key, "seed": s, "image": it["name"], "is_bg": it["is_bg"], "pred_px": int(pr.sum()),
                           "gt_px": int(it["gt"].sum()), "img_px": it["gt"].size, "tp": tp, "fp": fp, "fn": fn}
                    if it["raw"] is not None:
                        rtp, rfp, rfn = counts(pr, it["raw"])
                        row.update(raw_tp=rtp, raw_fp=rfp, raw_fn=rfn)
                    rows.append(row)
                if s == 0:
                    np.savez_compressed(qual / f"pred_seed0_{key}{args.tag}.npz",
                                        **{it["name"]: np.packbits(pr) for it, pr in zip(items, preds)},
                                        **{f"__shape__{it['name']}": np.array(pr.shape) for it, pr in zip(items, preds)})
                print(f"  {key:5s} seed {s}: {time.time() - t1:5.1f}s", flush=True)
    finally:
        cleanup()
    print(f"Tổng thời gian suy luận: {(time.time() - t0) / 60:.1f} phút")
    df = pd.DataFrame(rows)
    df.to_csv(f10, index=False, encoding="utf-8-sig")
    tong_hop(df, args.tag)


if __name__ == "__main__":
    main()
