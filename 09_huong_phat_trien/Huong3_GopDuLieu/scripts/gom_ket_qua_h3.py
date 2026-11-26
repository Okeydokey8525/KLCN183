"""Hướng 3, bước 5: gom 10 zip kết quả Kaggle (YOLO26s-seg, TSVM_FP32; seed 0-9; 24/11 gom thêm seed 5-9) vào ket_qua/.

- Giải nén mỗi zip vào ket_qua/<tên run>/ (bỏ qua nếu thư mục đã có).
- Chuyển zip gốc vào ket_qua/zip_tai_ve/ (chuyển, không xóa).
- Ghi ket_qua/bang_test_tung_seed.csv (số đo test do notebook tính, đọc từ test_metrics.json, không nhập tay)
  và in thống kê ghép cặp theo seed cho endpoint chính mask mAP50-95 (chỉ trên các seed có đủ cả hai mô hình).
"""
from __future__ import annotations

import csv
import io
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy import stats

H3 = Path(__file__).resolve().parent.parent
KQ = H3 / "ket_qua"
ZIPS = KQ / "zip_tai_ve"
CHI_SO = ["metrics/mAP50-95(M)", "metrics/mAP50(M)", "metrics/precision(M)", "metrics/recall(M)",
          "metrics/mAP50-95(B)", "metrics/mAP50(B)"]


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    ZIPS.mkdir(parents=True, exist_ok=True)
    for z in sorted(H3.glob("runs_H3_Gop_3tap_*.zip")):
        shutil.move(str(z), str(ZIPS / z.name))
    rows = []
    for z in sorted(ZIPS.glob("runs_H3_Gop_3tap_*.zip")):
        with zipfile.ZipFile(z) as zf:
            run = zf.namelist()[0].split("/")[0]
            if not (KQ / run).exists():
                zf.extractall(KQ)
        d = KQ / run
        m = json.loads((d / "test_metrics.json").read_text(encoding="utf-8"))
        with open(d / "results.csv", encoding="utf-8") as f:
            r = list(csv.DictReader(f))
        val = [float(x["metrics/mAP50-95(M)"]) for x in r]
        # Giờ huấn luyện lấy từ cột time (giây, cộng dồn) của results.csv; lượt chạy tiếp có time đặt lại về 0,
        # nên cộng giá trị cuối của từng đoạn. Không gồm thời gian đánh giá test, chép dữ liệu.
        tg = [float(x["time"]) for x in r]
        giay = sum(a for a, b in zip(tg, tg[1:]) if b < a) + tg[-1]
        rows.append({"model": m["model"], "seed": m["seed"], "so_epoch": len(r),
                     "dung_an_toan": (d / "CHUA_XONG.txt").exists(), "chay_tiep": bool(m.get("chay_tiep", False)),
                     "gio_huan_luyen": round(giay / 3600, 2),
                     "val_maskmAP50_95_max": round(max(val), 4), "epoch_val_max": int(np.argmax(val)) + 1,
                     **{k.replace("metrics/", "test_"): round(m[k], 4) for k in CHI_SO}})
    with open(KQ / "bang_test_tung_seed.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(r)
    y = {r["seed"]: r for r in rows if r["model"] != "TSVM_FP32"}
    t = {r["seed"]: r for r in rows if r["model"] == "TSVM_FP32"}
    seeds = sorted(set(y) & set(t))
    for k in ["test_mAP50-95(M)", "test_mAP50(M)", "test_mAP50-95(B)"]:
        a = np.array([y[s][k] for s in seeds]); b = np.array([t[s][k] for s in seeds]); d = b - a
        ci = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
        p_t = stats.ttest_rel(b, a).pvalue
        p_w = stats.wilcoxon(b, a).pvalue
        print(f"\n{k}: YOLO {a.mean():.4f} ± {a.std(ddof=1):.4f} | TSVM {b.mean():.4f} ± {b.std(ddof=1):.4f} | "
              f"Δ {d.mean():+.4f} [{d.mean() - ci:+.4f}; {d.mean() + ci:+.4f}] | p_t {p_t:.3f} | p_W {p_w:.3f} | "
              f"d_z {d.mean() / d.std(ddof=1):.2f} | TSVM thắng {int((d > 0).sum())}/{len(d)} | "
              f"Brown–Forsythe p {stats.levene(a, b, center='median').pvalue:.3f} (thăm dò) | seed {seeds}")


if __name__ == "__main__":
    main()
