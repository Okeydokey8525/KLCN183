"""Hướng 3: đánh giá theo pixel cho YOLO26s-seg và TSVM (seed 0–9) trên TẬP TEST gộp và 3 bộ dữ liệu ngoài.

Giao thức giống Hướng 2 (Huong2_Chia3Tap/danh_gia/danh_gia_test_h2.py) và Bảng 4.8 của báo cáo:
- YOLO: imgsz 640, conf 0,25, iou 0,7, retina_masks=True, half=False, FP32; gộp mọi mặt nạ của ảnh thành một.
- Checkpoint: best.pt (đã chọn trên val khi huấn luyện). Tập test và bộ ngoài chỉ dùng để đánh giá.
- Endpoint chính (chọn trước): Dice theo pixel trên TẬP TEST, trung bình trên ảnh polyp. Các chỉ số khác là phụ,
  mang tính thăm dò (không hiệu chỉnh so sánh bội).

Tập đánh giá:
- test: 522 ảnh (448 polyp, 74 âm tính) của du_lieu/H3_Gop_3tap. Nhãn chính là polygon YOLO vẽ ở độ phân giải ảnh
  (giống H2); nhãn phụ là mặt nạ masks/test/*.png (> 127). Ảnh âm tính xác định theo kiem_tra_nhan.csv (loai = am_tinh)
  và phải có nhãn rỗng. Nguồn ảnh lấy từ tiền tố tên (KV Kvasir, PG PolypGen, SW/SN PolypDB Simula WLI/NBI,
  KA Karolinska, CE cecum, PGR/PGS PolypGen âm tính).
- clinicdb (612), colondb (380), etis (196): ảnh video khác nguồn, không có trong train/val/test; mọi ảnh có polyp.
  Đường dẫn và cách đọc giống Test_Mau/08_du_lieu_thuc_te/scripts/danh_gia_du_lieu_ngoai.py.
  PolypGen KHÔNG dùng làm bộ ngoài vì đã nằm trong dữ liệu gộp.

Chỉ đọc dữ liệu, checkpoint và Test_Mau (mã mô hình nạp qua junction tạm). Ghi vào ../ket_qua/danh_gia/:
- bang/tung_luot/<tap>_<model>_s<seed>.csv: số đếm pixel từng ảnh, mỗi lượt một tệp; chạy lại bỏ qua lượt đã có.
- bang/11_tung_seed.csv, bang/12_thong_ke.csv, bang/14_theo_nguon_test.csv; du_lieu_dinh_tinh/ (mặt nạ dự đoán seed 0).

    python danh_gia_h3.py --kiem-tra                      # KHÔNG dùng GPU: kiểm dữ liệu, checkpoint, ước tính thời gian
    python danh_gia_h3.py --seeds 0 --models yolo --sets etis   # chạy thử 1 lượt (cần GPU)
    python danh_gia_h3.py                                  # 20 checkpoint × 4 tập (cần GPU)
    python danh_gia_h3.py --chi-tong-hop                   # chỉ tổng hợp lại từ tung_luot/
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
H3 = HERE.parent                                              # Huong3_GopDuLieu
DL = H3.parents[1]                                            # DeepLearning
TEST_MAU = DL / "Test_Mau"
FORK = TEST_MAU / "02_ma_nguon" / "ultralytics_TSVM_FP32"
DATA = H3 / "du_lieu" / "H3_Gop_3tap"
KQ = H3 / "ket_qua"
OUT = KQ / "danh_gia" / "bang"
LUOT = OUT / "tung_luot"
QUAL = KQ / "danh_gia" / "du_lieu_dinh_tinh"
DATASET = DL / "Dataset"
MODELS = {
    "yolo": ("YOLO26s-seg", lambda s: KQ / f"H3_Gop_3tap_YOLO26s_seg_s{s}"),
    "tsvm": ("YOLO26s-seg + TSVM", lambda s: KQ / f"H3_Gop_3tap_TSVM_FP32_s{s}"),
}
NGOAI = {
    "clinicdb": ("CVC-ClinicDB", DATASET / "CVC-ClinicDB" / "PNG" / "Original",
                 DATASET / "CVC-ClinicDB" / "PNG" / "Ground Truth", 612),
    "colondb": ("CVC-ColonDB", DATASET / "CVC-ColonDB_data" / "CVC-ColonDB" / "images",
                DATASET / "CVC-ColonDB_data" / "CVC-ColonDB" / "masks", 380),
    "etis": ("ETIS-LaribPolypDB", DATASET / "ETIS-Larib PolypDB" / "images", DATASET / "ETIS-Larib PolypDB" / "masks", 196),
}
SETS = ["test", *NGOAI]
TEN_TAP = {"test": "Test gộp H3", **{k: v[0] for k, v in NGOAI.items()}}
METRICS = ["dice", "iou", "precision", "recall", "dice_micro", "iou_micro", "no_pred_rate", "bg_fp_image_rate",
           "raw_dice", "raw_iou", "map50_95_mask", "map50_mask", "map50_95_box"]
NGUON = {"KV": "Kvasir-SEG", "PG": "PolypGen", "SW": "PolypDB Simula WLI", "SN": "PolypDB Simula NBI",
         "KA": "PolypDB Karolinska", "CE": "Cecum (âm tính)", "PGR": "PolypGen âm tính", "PGS": "PolypGen chuỗi âm tính"}


def doc(p, flag):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), flag)


def load_test():
    nhan = pd.read_csv(DATA / "kiem_tra_nhan.csv", encoding="utf-8-sig")
    loai = dict(zip(nhan[nhan.tap == "test"].ten_moi, nhan[nhan.tap == "test"].loai))
    items = []
    for p in sorted((DATA / "images" / "test").glob("*.jpg")):
        img = doc(p, cv2.IMREAD_COLOR)
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
        is_bg = loai[p.name] == "am_tinh"
        assert is_bg == (not gt.any()), (p.name, loai[p.name], int(gt.sum()))
        m = doc(DATA / "masks" / "test" / f"{p.stem}.png", cv2.IMREAD_GRAYSCALE)
        assert m is not None and m.shape == (h, w), (p.name, None if m is None else m.shape, (h, w))
        items.append({"name": p.name, "bgr": img, "gt": gt.astype(bool), "raw": None if is_bg else m > 127,
                      "is_bg": is_bg, "nguon": p.stem.split("_")[0]})
    assert len(items) == 522 and sum(i["is_bg"] for i in items) == 74, (len(items), sum(i["is_bg"] for i in items))
    return items


def load_ngoai(key):
    _, img_dir, mask_dir, n = NGOAI[key]
    items = []
    for p in sorted(img_dir.glob("*.png"), key=lambda q: int(q.stem) if q.stem.isdigit() else q.stem):
        img, m = doc(p, cv2.IMREAD_COLOR), doc(mask_dir / p.name, cv2.IMREAD_GRAYSCALE)
        assert img is not None and m is not None and m.shape == img.shape[:2], p
        items.append({"name": p.name, "bgr": img, "gt": m > 127, "raw": None, "is_bg": False, "nguon": key})
    assert len(items) == n, (key, len(items))
    return items


def load_set(key):
    return load_test() if key == "test" else load_ngoai(key)


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


def tom_tat(g: pd.DataFrame) -> dict:
    """Chỉ số của một nhóm ảnh (một tập, một mô hình, một seed)."""
    pol, bg = g[~g.is_bg], g[g.is_bg]
    r = {"n_polyp": len(pol), "n_am_tinh": len(bg)}
    if not len(pol):                                          # nhóm chỉ có ảnh âm tính (CE, PGR, PGS)
        cap = []
    else:
        cap = [("", ("tp", "fp", "fn"))] + ([("raw_", ("raw_tp", "raw_fp", "raw_fn"))] if "raw_tp" in pol
                                             and pol["raw_tp"].notna().all() else [])
    for pre, (a, b, c) in cap:
        m = np.array([per_image_metrics(x, y, z) for x, y, z in zip(pol[a], pol[b], pol[c])])
        r[f"{pre}dice"], r[f"{pre}iou"], r[f"{pre}precision"], r[f"{pre}recall"] = np.nanmean(m, 0)
        TP, FP, FN = pol[a].sum(), pol[b].sum(), pol[c].sum()
        r[f"{pre}dice_micro"] = 2 * TP / (2 * TP + FP + FN)
        r[f"{pre}iou_micro"] = TP / (TP + FP + FN)
    r["no_pred_rate"] = float((pol.pred_px == 0).mean()) if len(pol) else np.nan
    r["bg_fp_image_rate"] = float((bg.pred_px > 0).mean()) if len(bg) else np.nan
    return r


def per_seed(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (tap, key, s), g in df.groupby(["tap", "model", "seed"], sort=False):
        r = {"tap": tap, "model": key, "seed": int(s), **tom_tat(g)}
        if tap == "test":
            tm = json.loads((MODELS[key][1](s) / "test_metrics.json").read_text(encoding="utf-8"))
            r["map50_95_mask"] = tm["metrics/mAP50-95(M)"]
            r["map50_mask"] = tm["metrics/mAP50(M)"]
            r["map50_95_box"] = tm["metrics/mAP50-95(B)"]
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


def so_sanh(sd: pd.DataFrame, nhom: list[str]) -> list[dict]:
    """TSVM (a) so với YOLO (b), ghép theo seed, cho từng nhóm (tập hoặc tập + nguồn)."""
    rows = []
    if sd.empty:  # chạy thử không có tập test
        return rows
    for k, g in sd.groupby(nhom, sort=False):
        k = k if isinstance(k, tuple) else (k,)
        t, y = g[g.model == "tsvm"].set_index("seed"), g[g.model == "yolo"].set_index("seed")
        common = t.index.intersection(y.index)
        if len(common) < 3:
            continue
        for m in METRICS:
            if m in g.columns and g[m].notna().all():
                rows.append({**dict(zip(nhom, k)), "so_sanh": "H3: TSVM - YOLO", "metric": m,
                             **paired(t.loc[common, m], y.loc[common, m])})
    return rows


def doc_tung_luot() -> pd.DataFrame:
    fs = sorted(LUOT.glob("*.csv"))
    assert fs, f"Chưa có lượt nào trong {LUOT}"
    return pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in fs], ignore_index=True)


def tong_hop():
    df = doc_tung_luot()
    sd = per_seed(df)
    sd.to_csv(OUT / "11_tung_seed.csv", index=False, encoding="utf-8-sig")
    st = pd.DataFrame(so_sanh(sd, ["tap"]))
    st.to_csv(OUT / "12_thong_ke.csv", index=False, encoding="utf-8-sig")
    # Theo nguồn ảnh trong tập test (thăm dò; nhóm nhỏ như KA 4 ảnh chỉ để tham khảo)
    t = df[df.tap == "test"]
    rows = [{"tap": "test", "nguon": ng, "model": key, "seed": int(s), **tom_tat(g)}
            for (ng, key, s), g in t.groupby(["nguon", "model", "seed"], sort=False)]
    sn = pd.DataFrame(rows, columns=None if rows else ["tap", "nguon", "model", "seed"])
    st_ng = pd.DataFrame(so_sanh(sn, ["tap", "nguon"]))
    sn.to_csv(OUT / "14_theo_nguon_test.csv", index=False, encoding="utf-8-sig")
    st_ng.to_csv(OUT / "15_thong_ke_theo_nguon_test.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 250, "display.max_rows", 300):
        print(sd.groupby(["tap", "model"])[["dice", "iou", "precision", "recall", "no_pred_rate", "bg_fp_image_rate"]]
              .agg(["mean", "std"]).round(4).to_string())
        if len(st):
            print(st[st.metric.isin(["dice", "iou", "recall", "no_pred_rate", "bg_fp_image_rate", "map50_95_mask"])][
                ["tap", "metric", "mean_a", "sd_a", "mean_b", "sd_b", "delta", "ci95_low", "ci95_high",
                 "paired_t_p", "wilcoxon_p", "cohen_dz", "a_higher", "n_seeds"]].to_string(index=False))
        if len(st_ng):
            print(st_ng[st_ng.metric == "dice"][["nguon", "mean_a", "mean_b", "delta", "paired_t_p", "a_higher"]]
                  .to_string(index=False))


def kiem_tra(args):
    """Không dùng GPU, không nạp mô hình: kiểm checkpoint, đọc đủ dữ liệu, ước tính thời gian."""
    thieu = [str(MODELS[k][1](s)) for k in args.models for s in args.seeds
             if not (MODELS[k][1](s) / "weights" / "best.pt").exists() or not (MODELS[k][1](s) / "test_metrics.json").exists()]
    print(f"Checkpoint: {len(args.models) * len(args.seeds) - len(thieu)}/{len(args.models) * len(args.seeds)} có best.pt "
          f"và test_metrics.json" + (f"; THIẾU: {thieu}" if thieu else ""))
    tong = 0
    for key in args.sets:
        t0 = time.time()
        items = load_set(key)
        cnt = pd.Series([it["nguon"] for it in items]).value_counts().to_dict()
        kt = pd.Series([f"{it['gt'].shape[1]}x{it['gt'].shape[0]}" for it in items]).value_counts()
        print(f"[{key}] {len(items)} ảnh ({sum(not i['is_bg'] for i in items)} polyp, {sum(i['is_bg'] for i in items)} âm tính), "
              f"đọc {time.time() - t0:.1f}s | nguồn {cnt} | kích thước hay gặp {kt.head(3).to_dict()}")
        if key == "test":
            d = [2 * (i["gt"] & i["raw"]).sum() / (i["gt"].sum() + i["raw"].sum()) for i in items if not i["is_bg"]]
            print(f"   polygon so với mặt nạ gốc: Dice TB {np.mean(d):.4f}, nhỏ nhất {np.min(d):.4f}")
        tong += len(items)
        del items
    da_co = len(list(LUOT.glob("*.csv"))) if LUOT.exists() else 0
    luot = len(args.models) * len(args.seeds) * len(args.sets)
    # H2: 20 checkpoint × 160 ảnh mất 1,2 phút ≈ 22 ms/ảnh; TSVM chậm hơn YOLO ~1,5 lần, ảnh ETIS lớn hơn
    print(f"Tổng {tong} ảnh × {len(args.models) * len(args.seeds)} checkpoint = {tong * len(args.models) * len(args.seeds):,} "
          f"lần suy luận; {luot} lượt, đã có {da_co}. Ước tính 12–20 phút GPU.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=list(range(10)))
    ap.add_argument("--models", nargs="*", default=["yolo", "tsvm"], choices=list(MODELS))
    ap.add_argument("--sets", nargs="*", default=SETS, choices=SETS)
    ap.add_argument("--chi-tong-hop", action="store_true")
    ap.add_argument("--kiem-tra", action="store_true", help="không dùng GPU")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.kiem_tra:
        kiem_tra(args)
        return
    LUOT.mkdir(parents=True, exist_ok=True)
    if args.chi_tong_hop:
        tong_hop()
        return
    import torch
    assert torch.cuda.is_available(), "Cần GPU (C:\\venv_kl_gpu\\Scripts\\python.exe)"
    dev = "cuda:0"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    cleanup = import_fork()
    import ultralytics
    print("ultralytics:", ultralytics.__version__, ultralytics.__file__)
    QUAL.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        for tap in args.sets:
            viec = [(k, s) for k in args.models for s in args.seeds if not (LUOT / f"{tap}_{k}_s{s}.csv").exists()]
            if not viec:
                print(f"[{tap}] đã đủ, bỏ qua")
                continue
            items = load_set(tap)
            print(f"[{tap}] {len(items)} ảnh, {len(viec)} lượt cần chạy", flush=True)
            for key, s in viec:
                t1 = time.time()
                preds = predict_yolo(MODELS[key][1](s) / "weights" / "best.pt", items, dev)
                torch.cuda.empty_cache()
                rows = []
                for it, pr in zip(items, preds):
                    tp, fp, fn = counts(pr, it["gt"])
                    row = {"tap": tap, "model": key, "seed": s, "image": it["name"], "nguon": it["nguon"],
                           "is_bg": it["is_bg"], "pred_px": int(pr.sum()), "gt_px": int(it["gt"].sum()),
                           "img_px": it["gt"].size, "tp": tp, "fp": fp, "fn": fn}
                    if it["raw"] is not None:
                        rtp, rfp, rfn = counts(pr, it["raw"])
                        row.update(raw_tp=rtp, raw_fp=rfp, raw_fn=rfn)
                    rows.append(row)
                if s == 0:
                    np.savez_compressed(QUAL / f"pred_seed0_{tap}_{key}.npz",
                                        **{it["name"]: np.packbits(pr) for it, pr in zip(items, preds)},
                                        **{f"__shape__{it['name']}": np.array(pr.shape) for it, pr in zip(items, preds)})
                tmp = LUOT / f"{tap}_{key}_s{s}.csv.tmp"
                pd.DataFrame(rows).to_csv(tmp, index=False, encoding="utf-8-sig")
                tmp.replace(LUOT / f"{tap}_{key}_s{s}.csv")       # ghi xong mới đổi tên: lượt dở không bị tính
                print(f"  {tap:8s} {key:5s} seed {s}: {time.time() - t1:5.1f}s", flush=True)
            del items
    finally:
        cleanup()
    print(f"Tổng thời gian suy luận: {(time.time() - t0) / 60:.1f} phút")
    tong_hop()


if __name__ == "__main__":
    main()
