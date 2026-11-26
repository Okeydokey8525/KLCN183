"""Chép KẾT QUẢ các hướng phát triển (H1–H4) sang thư mục đích trong repo git (nhánh riêng), bỏ dữ liệu ảnh, trọng số, bản sao mã gốc.
Chạy thử:  python chep_sang_nhanh_git.py            (chỉ liệt kê, không chép)
Chép thật: python chep_sang_nhanh_git.py <thư_mục_đích>
"""
import os, shutil, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")
GOC = Path(__file__).resolve().parent
BO_THU_MUC = {"trong_so_tai_ve", "anh_gom", "du_lieu", "chon_du_lieu", "ma_nguon_goc", "__pycache__", ".ipynb_checkpoints",
              "1_dataset", "weights", "_cu"}
BO_TEN = {"kaggle.json", "GhiChu.txt", "CREDENTIALS.md", "RUNBOOK.md"}
GIU_DUOI = {".md", ".py", ".ipynb", ".csv", ".json", ".yaml", ".yml", ".txt"}
HINH = {".png", ".pdf", ".svg"}
TOI_DA = 20e6   # một tệp > 20 MB thì bỏ (GitHub cảnh báo > 50 MB)


def chon():
    out = []
    for r, ds, fs in os.walk(GOC):
        ds[:] = [d for d in ds if d not in BO_THU_MUC]
        rp = Path(r)
        la_luot = (rp / "results.csv").exists()      # thư mục một lượt huấn luyện Ultralytics
        for f in fs:
            p = rp / f; e = p.suffix.lower()
            if f in BO_TEN or ".bak" in f or p.stat().st_size > TOI_DA:
                continue
            if la_luot and f not in {"results.csv", "args.yaml", "gio.txt", "tom_tat_moc3.json", "results.png"}:
                continue
            if e in GIU_DUOI or e in HINH:
                out.append(p)
    return out


ds = chon()
tong = sum(p.stat().st_size for p in ds)
print(f"{len(ds)} tệp, {tong / 1e6:.1f} MB")
lon = sorted(ds, key=lambda p: -p.stat().st_size)[:8]
for p in lon:
    print(f"  {p.stat().st_size / 1e6:6.1f} MB  {p.relative_to(GOC)}")
if len(sys.argv) > 1:
    dich = Path(sys.argv[1])
    for p in ds:
        q = dich / p.relative_to(GOC); q.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(p, q)
    print("Đã chép sang", dich)
