"""Hướng 3, bước 1a: gom mọi ảnh ứng viên vào một thư mục, thu nhỏ về cạnh dài 640 để soi bằng ứng dụng tìm ảnh gần giống.

Chỉ đọc ảnh gốc (Dataset/, Test_Mau/), ghi bản sao đã thu nhỏ vào ../anh_gom/. Không đụng ảnh gốc.
- Ảnh có cạnh dài > 640 được thu nhỏ (LANCZOS, giữ tỉ lệ) về cạnh dài 640. Ảnh nhỏ hơn giữ nguyên kích thước, không phóng to.
- Lưu JPEG chất lượng 95.
- Tên mới có tiền tố nguồn để biết ảnh từ đâu; bảng anh_gom/danh_sach_anh.csv nối tên mới với đường dẫn gốc.

anh_gom/polyp/    KV_ (Kvasir-SEG), PG_C1.._C6_ (PolypGen ảnh đơn có polyp), SW_ (Simula WLI, đã bỏ 25 ảnh trùng BG20),
                  SN_ (Simula NBI), KA_ (Karolinska)
anh_gom/am_tinh/  CE_ (Kvasir normal-cecum, 1.000 ảnh; 200 ảnh đã nằm trong BG20 ghi ở cột da_dung_BG20),
                  PGR_C?_ (PolypGen ảnh đơn mặt nạ rỗng), PGS_seqNN_ (PolypGen chuỗi âm tính, mọi khung)

Chạy: C:\\venv_kl_gpu\\Scripts\\python.exe gom_anh_640.py   (dừng nếu anh_gom/ đã có ảnh, để không ghi đè)
"""
from __future__ import annotations

import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

HERE = Path(__file__).resolve().parent
H3 = HERE.parent
DL = H3.parent.parent                                  # DeepLearning/
BANG = H3 / "kiem_tra_du_lieu" / "bang"
OUT = H3 / "anh_gom"
CANH = 640
PG = DL / "Dataset" / "PolypGen2021_MultiCenterData_v3" / "PolypGen2021_MultiCenterData_v3"
CECUM = DL / "Test_Mau" / "01_du_lieu" / "tao_BG20" / "normal-cecum" / "normal-cecum"
TIEN_TO_PDB = {"Simula/WLI": "SW", "Simula/NBI": "SN", "Karolinska/WLI": "KA"}


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def mask_rong(img: Path, c: str) -> bool:
    """PolypGen: mặt nạ cùng tên (thêm _mask) trong masks_Cx; rỗng nếu không pixel nào > 127."""
    d = PG / f"data_{c}" / f"masks_{c}"
    s = img.stem.rstrip("]")         # C3_EndoCV2021_00489].jpg: tên ảnh gốc có dấu ] thừa
    ung = [q for q in d.iterdir() if q.stem in (s + "_mask", s)]
    if len(ung) != 1:
        raise RuntimeError(f"không tìm đúng 1 mặt nạ cho {img.name}: {ung}")
    return int(np.asarray(Image.open(ung[0]).convert("L")).max()) <= 127


def danh_sach() -> pd.DataFrame:
    p = pd.read_csv(BANG / "01_tung_anh.csv", encoding="utf-8-sig")
    loai = pd.read_csv(BANG / "06_anh_loai_vi_trung_BG20.csv", encoding="utf-8-sig")
    bo_ra = set(loai.anh_loai)
    rows = []
    for r in p.itertuples():
        f = Path(r.duong_dan)
        if r.bo == "BG20_train_cu" and r.nhom == "kvasir":
            rows.append(("polyp", "KV", "Kvasir-SEG", "", f))
        elif r.bo == "PolypGen_don":
            rong = mask_rong(f, r.nhom)
            rows.append(("am_tinh" if rong else "polyp", f"PGR_{r.nhom}" if rong else f"PG_{r.nhom}",
                         "PolypGen ảnh đơn" + (" (mặt nạ rỗng)" if rong else ""), r.nhom, f))
        elif r.bo == "PolypDB" and r.nhom in TIEN_TO_PDB:
            nhom = r.nhom
            # 5 ảnh Simula nằm trong thư mục NBI nhưng bảng 01 ghi nhóm Simula/WLI (trùng tên với 5 ảnh WLI,
            # khác nội dung). Xếp theo thư mục thật: NBI.
            if nhom == "Simula/WLI" and f.parent.parent.name == "NBI":
                nhom = "Simula/NBI"
            if nhom == "Simula/WLI" and f.name in bo_ra:
                continue
            rows.append(("polyp", TIEN_TO_PDB[nhom], "PolypDB " + nhom, nhom, f))
    bg20 = {Path(x).name.removeprefix("bg_") for x in p[(p.bo == "BG20_train_cu") & (p.nhom == "nen")].duong_dan}
    for f in sorted(CECUM.glob("*.jpg")):
        rows.append(("am_tinh", "CE", "Kvasir normal-cecum" + (" (đã dùng trong BG20)" if f.name in bg20 else ""), "", f))
    for sd in sorted((PG / "sequenceData" / "negativeOnly").iterdir()):
        for f in sorted(sd.glob("*.jpg")):
            rows.append(("am_tinh", f"PGS_{sd.name.removesuffix('_neg')}", "PolypGen chuỗi âm tính", sd.name, f))
    d = pd.DataFrame(rows, columns=["thu_muc", "tien_to", "nguon", "nhom", "duong_dan_goc"])
    d["ten_moi"] = d.tien_to + "_" + d.duong_dan_goc.map(lambda x: Path(x).stem.replace(" ", "_")) + ".jpg"
    assert d.ten_moi.is_unique, d[d.ten_moi.duplicated(keep=False)]
    return d


def lam_mot(r):
    src = Path(r.duong_dan_goc)
    with Image.open(src) as im:
        im = im.convert("RGB")
        w, h = im.size
        s = CANH / max(w, h)
        if s < 1:
            im = im.resize((round(w * s), round(h * s)), Image.LANCZOS)
        im.save(OUT / r.thu_muc / r.ten_moi, quality=95)
        return w, h, im.size[0], im.size[1], md5(src)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if OUT.exists() and any(OUT.rglob("*.jpg")):
        sys.exit(f"{OUT} đã có ảnh, dừng để không ghi đè.")
    d = danh_sach()
    for t in ("polyp", "am_tinh"):
        (OUT / t).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        kq = list(ex.map(lam_mot, d.itertuples()))
    d[["w_goc", "h_goc", "w_moi", "h_moi", "md5_goc"]] = kq
    d["da_dung_BG20"] = d.nguon.str.contains("đã dùng trong BG20") | (d.tien_to == "KV")
    d.to_csv(OUT / "danh_sach_anh.csv", index=False, encoding="utf-8-sig")
    print(d.groupby(["thu_muc", "nguon"]).agg(so_anh=("ten_moi", "size"), canh_dai_moi_max=("w_moi", "max")).to_string())
    print("Tổng:", len(d), "| md5 gốc lặp:", int(d.md5_goc.duplicated().sum()))
    print("Cạnh dài mới: min", int(d[["w_moi", "h_moi"]].max(axis=1).min()), "max", int(d[["w_moi", "h_moi"]].max(axis=1).max()))


if __name__ == "__main__":
    main()
