"""Tổng hợp 20 seed TSVM kernel CUDA trên dữ liệu ngoài + 800 ảnh manh tràng (công thức chép đúng summarize()/tong_hop()
của 08_du_lieu_thuc_te), rồi so ghép cặp với TSVM PyTorch và YOLO26s-seg của báo cáo (Mục 4.7).
Ghi ket_qua_moc3/du_lieu_ngoai/{01_cuda_tung_seed.csv, 02_so_sanh_ghep_cap.csv, 03_cecum_so_sanh.csv}"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
H4 = Path(__file__).resolve().parents[1]
TM = H4.parents[1] / "Test_Mau"
sys.path.insert(0, str(TM / "00_tai_lieu_tong_quan" / "scripts"))
import danh_gia_pixel as dg  # noqa: E402

D = H4 / "ket_qua_moc3" / "du_lieu_ngoai"
P = TM / "00_tai_lieu_tong_quan" / "bang" / "pixel_640"

# --- từng seed của bản CUDA (giống summarize())
df = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in sorted((D / "tung_luot").glob("*_tsvm_s*.csv"))], ignore_index=True)
df = pd.concat([df, df[df.dataset.str.startswith("pg_")].assign(dataset="polypgen")], ignore_index=True)
assert (df.groupby("dataset").seed.nunique() == 20).all()
rows = []
for (key, s), ga in df.groupby(["dataset", "seed"]):
    g, bg = ga[ga.gt_px > 0], ga[ga.gt_px == 0]
    m = np.array([dg.per_image_metrics(a, b, c) for a, b, c in zip(g.tp, g.fp, g.fn)])
    r = {"dataset": key, "model": "tsvm_cuda", "seed": s, "n_images": len(g), "n_no_polyp": len(bg),
         "bg_fp_image_rate": float((bg.pred_px > 0).mean()) if len(bg) else np.nan}
    r["dice"], r["iou"], r["precision"], r["recall"] = np.nanmean(m, 0)
    TP, FP, FN = g.tp.sum(), g.fp.sum(), g.fn.sum()
    r["dice_micro"] = 2 * TP / (2 * TP + FP + FN); r["iou_micro"] = TP / (TP + FP + FN)
    r["no_pred_rate"] = float((g.pred_px == 0).mean())
    rows.append(r)
CU = pd.DataFrame(rows)
CU.to_csv(D / "01_cuda_tung_seed.csv", index=False, encoding="utf-8-sig")
BC = pd.read_csv(P / "du_lieu_ngoai" / "20seed" / "01_ngoai_tung_seed.csv", encoding="utf-8-sig")


def ghep(a, b):
    d = b - a; sd = d.std(ddof=1); ci = stats.t.ppf(0.975, len(d) - 1) * sd / np.sqrt(len(d))
    ok = sd > 0
    return {"doi_chieu": a.mean(), "cuda": b.mean(), "cuda_sd": b.std(ddof=1), "delta": d.mean(), "ci_lo": d.mean() - ci,
            "ci_hi": d.mean() + ci, "p_t": stats.ttest_rel(b, a).pvalue if ok else np.nan,
            "p_wilcoxon": stats.wilcoxon(b, a).pvalue if ok else np.nan, "cuda_cao_hon": int((d > 0).sum())}


THU_TU = ["clinicdb", "colondb", "etis", "polypgen", "pg_c1", "pg_c2", "pg_c3", "pg_c4", "pg_c5", "pg_c6"]
out = []
for ref in ("tsvm", "baseline"):
    for ds in THU_TU:
        a = BC[(BC.model == ref) & (BC.dataset == ds)].set_index("seed").loc[range(20)]
        b = CU[CU.dataset == ds].set_index("seed").loc[range(20)]
        assert (a.n_images == b.n_images).all()
        for m in ("dice", "iou", "recall", "precision", "dice_micro", "no_pred_rate", "bg_fp_image_rate"):
            if a[m].isna().all():
                continue
            out.append({"so_voi": ref, "dataset": ds, "metric": m, **ghep(a[m].to_numpy(), b[m].to_numpy())})
R = pd.DataFrame(out)
R.to_csv(D / "02_so_sanh_ghep_cap.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(R[R.metric == "dice"].round(4).to_string(index=False))
print(R[R.metric.isin(["no_pred_rate", "bg_fp_image_rate"]) & (R.so_voi == "tsvm")].round(4).to_string(index=False))

# --- 800 ảnh manh tràng (không ảnh nào gần trùng với ảnh nền train, nên fp_image_rate trên cả 800 ảnh)
c = pd.concat([pd.read_csv(f, encoding="utf-8-sig") for f in sorted((D / "nen_cecum_tung_luot").glob("cecum800_tsvm_s*.csv"))])
cs = c.groupby("seed").apply(lambda g: pd.Series({"fp_image_rate": (g.pred_px > 0).mean(), "n": len(g)}), include_groups=False)
assert len(cs) == 20 and (cs.n == 800).all()
BCc = pd.read_csv(P / "nen_cecum" / "06_nen_cecum_tung_seed.csv", encoding="utf-8-sig")
cr = []
for ref in ("tsvm", "baseline"):
    a = BCc[BCc.model == ref].set_index("seed").loc[range(20), "fp_image_rate"].to_numpy()
    cr.append({"so_voi": ref, "metric": "cecum800_fp_image_rate", **ghep(a, cs.loc[range(20), "fp_image_rate"].to_numpy())})
cr = pd.DataFrame(cr)
cr.to_csv(D / "03_cecum_so_sanh.csv", index=False, encoding="utf-8-sig")
print(cr.round(4).to_string(index=False))
