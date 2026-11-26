"""Chạy thử toàn bộ mã mốc 2 (h4_kiem_tra.py, phần A–F) trên laptop với kernel GIẢ LẬP (selective scan viết tay),
số lần đo nhỏ. Mục đích: bắt lỗi mã trước khi tốn giờ GPU Kaggle. Số đo ở đây KHÔNG dùng để kết luận tốc độ.

    C:\\venv_kl_gpu\\Scripts\\python.exe -X utf8 chay_thu_moc2_laptop.py
Kết quả ghi vào thư mục tạm, in đường dẫn cuối cùng.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
H4 = HERE.parent
DL = H4.parents[1]
TM = DL / "Test_Mau"
FORK = H4 / "ma_nguon" / "ultralytics_TSVM_CUDA"
KT_DIR = H4 / "Goi_Kaggle_H4" / "ma_kiem_tra"
DATA = TM / "01_du_lieu" / "Kvasir_YOLO_SEG_BG20"
W_BASE = TM / "03_ket_qua_huan_luyen" / "Baseline_YOLO26s_seg" / "Kvasir_BG20_Baseline_YOLO26s_seg_s0_w2" / "weights" / "best.pt"
W_TSVM = TM / "03_ket_qua_huan_luyen" / "TSVM_FP32" / "Kvasir_BG20_YOLO26s_seg_TSVM_FP32_s0_w2" / "weights" / "best.pt"


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    tmp = Path(tempfile.mkdtemp(prefix="h4_moc2_thu_"))
    subprocess.run(["cmd", "/c", "mklink", "/J", str(tmp / "ultralytics"), str(FORK)], check=True, capture_output=True)
    sys.path[:0] = [str(tmp), str(KT_DIR)]
    try:
        import ultralytics
        import ultralytics.nn.modules.topology_shape_vmamba as tsv
        import h4_kiem_tra as hk

        assert Path(ultralytics.__file__).resolve().parent == FORK.resolve()
        tsv._selective_scan_fn = hk.selective_scan_ref  # giả lập kernel
        for sub in ("images/val", "labels/val"):  # Test_Mau chỉ đọc: chép val ra thư mục tạm
            shutil.copytree(DATA / sub, tmp / "data" / sub)
        data_yaml = tmp / "data.yaml"
        data_yaml.write_text(yaml.safe_dump({"path": str(tmp / "data"), "train": "images/val", "val": "images/val",
                                             "nc": 1, "names": {0: "polyp"}}), encoding="utf-8")
        hk.chay_tat_ca(W_BASE, W_TSVM, tmp / "data" / "images" / "val", data_yaml, tmp / "out", nhanh=True)
        print("THƯ MỤC KẾT QUẢ CHẠY THỬ:", tmp / "out")
    finally:
        subprocess.run(["cmd", "/c", "rmdir", str(tmp / "ultralytics")], capture_output=True)


if __name__ == "__main__":
    main()
