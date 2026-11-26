"""So sánh ghép cặp 20 seed theo pixel trên tập test: TSVM kernel CUDA với TSVM PyTorch và YOLO26s-seg của báo cáo
(Bảng 4.8, Test_Mau/00_tai_lieu_tong_quan/bang/pixel_640/11_pixel_tung_seed.csv). Ghi ket_qua_moc3/pixel_test/13_so_sanh_ghep_cap.csv"""
import sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
H4 = Path(__file__).resolve().parents[1]
BC = pd.read_csv(H4.parents[1] / "Test_Mau/00_tai_lieu_tong_quan/bang/pixel_640/11_pixel_tung_seed.csv", encoding="utf-8-sig")
CU = pd.read_csv(H4 / "ket_qua_moc3/pixel_test/11_pixel_tung_seed.csv", encoding="utf-8-sig")
M = ["dice", "iou", "precision", "recall", "bg_fp_image_rate", "raw_dice"]
assert sorted(CU.seed) == list(range(20))
rows = []
for doi, ten in (("tsvm", "TSVM PyTorch"), ("baseline", "YOLO26s-seg")):
    ref = BC[BC.model == doi].set_index("seed").loc[range(20)]
    cu = CU.set_index("seed").loc[range(20)]
    for m in M:
        a, b = ref[m].to_numpy(), cu[m].to_numpy(); d = b - a
        ci = stats.t.ppf(0.975, 19) * d.std(ddof=1) / np.sqrt(20)
        try:
            pw = stats.wilcoxon(b, a).pvalue
        except ValueError:
            pw = np.nan
        rows.append({"so_voi": ten, "metric": m, "doi_chieu_mean": a.mean(), "doi_chieu_sd": a.std(ddof=1),
                     "cuda_mean": b.mean(), "cuda_sd": b.std(ddof=1), "delta": d.mean(), "ci_lo": d.mean() - ci,
                     "ci_hi": d.mean() + ci, "p_t": stats.ttest_rel(b, a).pvalue, "p_wilcoxon": pw,
                     "dz": d.mean() / d.std(ddof=1) if d.std() > 0 else np.nan, "cuda_cao_hon": int((d > 0).sum()),
                     "bang_nhau": int((d == 0).sum()), "p_levene": stats.levene(a, b, center="median").pvalue})
R = pd.DataFrame(rows)
R.to_csv(H4 / "ket_qua_moc3/pixel_test/13_so_sanh_ghep_cap.csv", index=False, encoding="utf-8-sig")
pd.set_option("display.width", 250)
print(R.round(4).to_string(index=False))
