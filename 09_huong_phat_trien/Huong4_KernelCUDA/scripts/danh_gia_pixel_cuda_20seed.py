"""Hướng 4: đánh giá theo pixel 20 best.pt TSVM bản kernel CUDA trên tập test (val BG20 160 ảnh = 120 ảnh test
chính thức của Kvasir-SEG + 40 ảnh nền), đúng giao thức Bảng 4.8 (gọi lại danh_gia_pixel.py của báo cáo, không sửa nó).
Suy luận trên laptop bằng nhánh PyTorch của fork báo cáo (best.pt dùng chung được giữa hai bản).
Kết quả: ket_qua_moc3/pixel_test/10–12_*.csv
Chạy: C:\\venv_kl_gpu\\Scripts\\python.exe -X utf8 danh_gia_pixel_cuda_20seed.py
"""
import glob, os, sys
from pathlib import Path

H4 = Path(__file__).resolve().parents[1]
SCR = H4.parents[1] / "Test_Mau" / "00_tai_lieu_tong_quan" / "scripts"
sys.path.insert(0, str(SCR))
import danh_gia_pixel as dp  # noqa: E402


def w(s):
    g = glob.glob(str(H4 / "ket_qua_moc3" / "tai_ve_kaggle" / "*" / f"Kvasir_BG20_YOLO26s_seg_TSVM_CUDA_s{s}_w2" / "weights" / "best.pt"))
    assert len(g) == 1, (s, g)
    return Path(g[0])


# Khóa "tsvm" được trỏ sang best.pt bản CUDA (script gốc chỉ nhận baseline/tsvm cho YOLO); ghi ra thư mục riêng
dp.MODELS["tsvm"] = ("YOLO26s-seg + TSVM (kernel CUDA)", w)
out = H4 / "ket_qua_moc3" / "pixel_test"
out.mkdir(parents=True, exist_ok=True)
sys.argv = [sys.argv[0], "--models", "tsvm", "--out", str(out)] + sys.argv[1:]
dp.main()
