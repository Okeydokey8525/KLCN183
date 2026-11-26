"""Hướng 4: đánh giá 20 best.pt TSVM kernel CUDA trên dữ liệu khác nguồn (CVC-ClinicDB, CVC-ColonDB, ETIS, PolypGen C1–C6)
và 800 ảnh manh tràng, đúng giao thức Mục 4.7 (gọi lại run() của 08_du_lieu_thuc_te/scripts, không sửa script gốc).
Kết quả từng lượt ghi vào ket_qua_moc3/du_lieu_ngoai/, KHÔNG ghi vào Test_Mau.
Chạy (GPU laptop): C:\\venv_kl_gpu\\Scripts\\python.exe -X utf8 danh_gia_ngoai_cuda_20seed.py --phan ngoai|cecum [--seeds 0]
"""
import argparse, glob, sys
from pathlib import Path

H4 = Path(__file__).resolve().parents[1]
TM = H4.parents[1] / "Test_Mau"
sys.path.insert(0, str(TM / "00_tai_lieu_tong_quan" / "scripts"))
sys.path.insert(0, str(TM / "08_du_lieu_thuc_te" / "scripts"))
import danh_gia_pixel as dg  # noqa: E402
import danh_gia_du_lieu_ngoai as ng  # noqa: E402
import danh_gia_nen_cecum as nc  # noqa: E402

OUT = H4 / "ket_qua_moc3" / "du_lieu_ngoai"


def w(s):
    g = glob.glob(str(H4 / "ket_qua_moc3" / "tai_ve_kaggle" / "*" / f"Kvasir_BG20_YOLO26s_seg_TSVM_CUDA_s{s}_w2" / "weights" / "best.pt"))
    assert len(g) == 1, (s, g)
    return Path(g[0])


# Script gốc chỉ nhận khóa baseline/tsvm cho YOLO: trỏ khóa "tsvm" sang best.pt bản CUDA, ghi ra thư mục riêng.
dg.MODELS["tsvm"] = ("YOLO26s-seg + TSVM (kernel CUDA)", w)
ng.LUOT = OUT / "tung_luot"
ng.QUAL = OUT / "du_lieu_dinh_tinh"
nc.LUOT = OUT / "nen_cecum_tung_luot"
for p in (ng.LUOT, ng.QUAL, nc.LUOT):
    p.mkdir(parents=True, exist_ok=True)
assert TM.resolve() not in ng.LUOT.resolve().parents

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", nargs="*", type=int, default=list(range(20)))
ap.add_argument("--phan", choices=["ngoai", "cecum"], required=True, help="mỗi tiến trình chỉ nạp fork một lần")
a = ap.parse_args()
sys.stdout.reconfigure(encoding="utf-8")
if a.phan == "ngoai":
    ng.run(argparse.Namespace(sets=list(ng.SETS), models=["tsvm"], seeds=a.seeds))
else:
    con_lai, _ = nc.danh_sach()
    nc.run(argparse.Namespace(models=["tsvm"], seeds=a.seeds), con_lai)
