"""Đánh giá theo pixel cho VM-UNet 640 (seed 0–4), cùng giao thức Bảng 4.8 và cùng cách làm với Polyp-PVT.

- Dựng mô hình bằng mã gốc VM-UNet, thay phép quét CUDA của mamba_ssm bằng bản PyTorch thuần
  (vmunet_quet_thuan.py), vì mamba_ssm không cài được trên Windows.
- Kiểm chứng đầu-cuối: từ cùng một lần suy luận, tính lại Val Dice đúng như notebook (ảnh 640, mặt nạ co
  NEAREST, ảnh nền đúng khi không báo pixel nào) rồi so với best_val_dice trong tom_tat.txt.
- Giao thức Bảng 4.8: logit (trước sigmoid) phóng song tuyến về kích thước gốc, sigmoid > 0,5; nhãn polygon ở
  độ phân giải gốc, thêm nhãn phụ Kvasir-SEG. Dữ liệu và hàm chỉ số lấy từ danh_gia_pixel_pvt.py.

Kết quả GHI vào ket_qua/danh_gia_pixel/ với tên MỚI (không ghi đè file 5 mô hình):
    10_pixel_tung_anh_vmunet.csv, 11_pixel_tung_seed_6model.csv, 12_pixel_thong_ke_6model.csv,
    13_kiem_chung_val_dice_vmunet.csv, du_lieu_dinh_tinh/pred_seed0_vmunet640.npz

    C:\\venv_kl_gpu\\Scripts\\python.exe danh_gia_pixel_vmunet.py [--seeds 0 --tag _thu] [--tu-bang]
"""
from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import stats

sys.dont_write_bytecode = True
import danh_gia_pixel_pvt as P                                    # noqa: E402
from vmunet_quet_thuan import tao_vmunet                          # noqa: E402

CKPT = lambda s: P.GOC / "ket_qua" / "VMUNet_640" / f"VMUNet_640_aug_s{s}" / f"vmunet_best_s{s}.pth"
TOM_TAT = lambda s: CKPT(s).parent / "tom_tat.txt"
OUT = P.OUT
NHAN = {**P.NHAN, "vmunet640": "VM-UNet 640"}
SO_VOI = ["polyppvt640", "pranet640", "tsvm", "baseline", "unet640"]


def du_doan(net, items, dev):
    """Trả về (mặt nạ ở kích thước gốc theo giao thức Bảng 4.8, Val Dice kiểu notebook của từng ảnh)."""
    preds, nb_dice = [], []
    with torch.no_grad():
        for it in items:
            h, w = it["gt"].shape
            rgb = cv2.cvtColor(it["bgr"], cv2.COLOR_BGR2RGB)
            x = cv2.resize(rgb, (P.SIZE, P.SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            x = (x - P.MEAN) / P.STD
            t = torch.from_numpy(x).permute(2, 0, 1)[None].to(dev)
            logit = net.vmunet(t)                                   # VMUNet.forward = sigmoid(self.vmunet(x))
            # Val Dice như notebook: so ở 640 với mặt nạ co NEAREST
            p640 = (torch.sigmoid(logit)[0, 0] > 0.5).cpu().numpy()
            g640 = cv2.resize(it["gt"].astype(np.uint8) * 255, (P.SIZE, P.SIZE), interpolation=cv2.INTER_NEAREST) > 127
            if g640.sum() == 0:
                nb_dice.append(1.0 if p640.sum() == 0 else 0.0)
            else:
                nb_dice.append(2.0 * (p640 & g640).sum() / (p640.sum() + g640.sum() + 1e-5))
            up = F.interpolate(logit, size=(h, w), mode="bilinear", align_corners=False)
            preds.append((torch.sigmoid(up)[0, 0] > 0.5).cpu().numpy())
    return preds, nb_dice


def thong_ke(sd):
    rows = []
    for key in sd.model.unique():
        g = sd[sd.model == key]
        for m in P.METRICS:
            rows.append({"model": key, "label": NHAN.get(key, key), "metric": m, "n_seeds": len(g),
                         "mean": g[m].mean(), "sd": g[m].std(ddof=1), "min": g[m].min(), "max": g[m].max()})
    v = sd[sd.model == "vmunet640"].set_index("seed")
    for other in [k for k in SO_VOI if k in set(sd.model)]:
        o = sd[sd.model == other].set_index("seed")
        common = v.index.intersection(o.index)
        if len(common) < 3:
            continue
        for m in P.METRICS:
            d = v.loc[common, m] - o.loc[common, m]
            ok = d.std() > 0
            rows.append({"model": f"vmunet640_vs_{other}", "metric": m, "n_seeds": len(common),
                         "mean": d.mean(), "sd": d.std(ddof=1),
                         "paired_t_p": stats.ttest_rel(v.loc[common, m], o.loc[common, m]).pvalue if ok else np.nan,
                         "wilcoxon_p": stats.wilcoxon(v.loc[common, m], o.loc[common, m]).pvalue if ok else np.nan,
                         "vmunet_higher": int((d > 0).sum())})
    return pd.DataFrame(rows)


def tong_hop(df_vm, tag):
    OUT.mkdir(parents=True, exist_ok=True)
    df_vm.to_csv(OUT / f"10_pixel_tung_anh_vmunet{tag}.csv", index=False, encoding="utf-8-sig")
    seeds = sorted(df_vm.seed.unique())
    pvt = pd.read_csv(OUT / "10_pixel_tung_anh_polyppvt.csv", encoding="utf-8-sig")
    cu = pd.read_csv(P.BANG_CU, encoding="utf-8-sig")
    cu = cu[cu.model.isin(["baseline", "tsvm", "pranet640", "unet640"])]
    allm = pd.concat([df_vm, pvt, cu], ignore_index=True)
    sd = P.theo_seed(allm[allm.seed.isin(seeds)])
    sd.to_csv(OUT / f"11_pixel_tung_seed_6model{tag}.csv", index=False, encoding="utf-8-sig")
    st = thong_ke(sd)
    st.to_csv(OUT / f"12_pixel_thong_ke_6model{tag}.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.float_format", "{:.4f}".format, "display.width", 220, "display.max_rows", 300):
        print(st[st.metric.isin(["dice", "iou", "precision", "recall", "bg_fp_image_rate", "raw_dice"])].to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="*", type=int, default=P.SEEDS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--tu-bang", action="store_true")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.tu_bang:
        df = pd.read_csv(OUT / f"10_pixel_tung_anh_vmunet{args.tag}.csv", encoding="utf-8-sig")
        tong_hop(df[df.seed.isin(args.seeds)], args.tag)
        return

    for p in [P.DATA, P.RAW_MASKS, P.BANG_CU, OUT / "10_pixel_tung_anh_polyppvt.csv"] + [CKPT(s) for s in args.seeds]:
        assert p.exists(), f"Thiếu: {p}"
    assert torch.cuda.is_available(), "Cần GPU (chạy bằng C:\\venv_kl_gpu\\Scripts\\python.exe)"
    dev = "cuda:0"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)

    items = P.load_val()
    print(f"Validation: {len(items)} ảnh ({sum(not i['is_bg'] for i in items)} polyp, {sum(i['is_bg'] for i in items)} nền)")
    qual = OUT / "du_lieu_dinh_tinh"
    qual.mkdir(parents=True, exist_ok=True)

    rows, kc, t0 = [], [], time.time()
    for s in args.seeds:
        t1 = time.time()
        net = tao_vmunet()
        net.load_state_dict(torch.load(CKPT(s), map_location="cpu", weights_only=True), strict=True)
        net = net.to(dev).eval()
        preds, nb = du_doan(net, items, dev)
        del net
        torch.cuda.empty_cache()
        tt = dict(kv.split("=") for kv in TOM_TAT(s).read_text().split("\n")[1].split())
        kc.append({"seed": s, "best_epoch": int(tt["best_epoch"]), "val_dice_notebook": float(tt["best_val_dice"]),
                   "val_dice_tinh_lai": float(np.mean(nb)), "chenh": float(np.mean(nb)) - float(tt["best_val_dice"])})
        for it, pr in zip(items, preds):
            tp, fp, fn = P.counts(pr, it["gt"])
            row = {"model": "vmunet640", "seed": s, "image": it["name"], "is_bg": it["is_bg"],
                   "pred_px": int(pr.sum()), "gt_px": int(it["gt"].sum()), "img_px": it["gt"].size,
                   "tp": tp, "fp": fp, "fn": fn}
            if it["raw"] is not None:
                rtp, rfp, rfn = P.counts(pr, it["raw"])
                row.update(raw_tp=rtp, raw_fp=rfp, raw_fn=rfn)
            rows.append(row)
        if s == 0:
            np.savez_compressed(qual / "pred_seed0_vmunet640.npz",
                                **{it["name"]: np.packbits(pr) for it, pr in zip(items, preds)},
                                **{f"__shape__{it['name']}": np.array(pr.shape) for it, pr in zip(items, preds)})
        print(f"  vmunet640 seed {s}: {time.time() - t1:5.1f}s | Val Dice notebook {kc[-1]['val_dice_notebook']:.4f}"
              f" | tính lại {kc[-1]['val_dice_tinh_lai']:.4f} (chênh {kc[-1]['chenh']:+.4f})", flush=True)
    print(f"Tổng thời gian suy luận: {(time.time() - t0) / 60:.1f} phút")
    pd.DataFrame(kc).to_csv(OUT / f"13_kiem_chung_val_dice_vmunet{args.tag}.csv", index=False, encoding="utf-8-sig")
    tong_hop(pd.DataFrame(rows), args.tag)


if __name__ == "__main__":
    main()
