"""Hướng 3, bước 3b: đổi tên tệp có ký tự Kaggle không nhận (']', '(', ')', ...) trong du_lieu/H3_Gop_3tap/.

Kaggle từ chối tải lên zip có tên 'PG_C3_C3_EndoCV2021_00489].jpg' (dấu ']' có sẵn trong tên gốc PolypGen).
Đổi tên ảnh, nhãn, mặt nạ cho cùng một stem; cập nhật kiem_tra_nhan.csv; ghi bảng đối chiếu doi_ten_kaggle.csv
(tên cũ = tên trong anh_gom/ và chon_du_lieu/danh_sach_chon.csv). Nội dung tệp không đổi.
Chạy lại an toàn: tệp đã đổi thì bỏ qua.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

H3 = Path(__file__).resolve().parent.parent
DATA = H3 / "du_lieu" / "H3_Gop_3tap"
TAPS = ["train", "val", "test"]


def ten_an_toan(stem: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    return re.sub(r"_+", "_", s).strip("_")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    bang_path = DATA / "doi_ten_kaggle.csv"
    cu = pd.read_csv(bang_path, encoding="utf-8-sig") if bang_path.exists() else pd.DataFrame(
        columns=["tap", "ten_cu", "ten_moi_kaggle"])
    rows = []
    for tap in TAPS:
        for img in sorted((DATA / "images" / tap).iterdir()):
            moi = ten_an_toan(img.stem)
            if moi == img.stem:
                continue
            for sub, duoi in (("images", img.suffix), ("labels", ".txt"), ("masks", ".png")):
                a, b = DATA / sub / tap / f"{img.stem}{duoi}", DATA / sub / tap / f"{moi}{duoi}"
                if b.exists():
                    sys.exit(f"Trùng tên đích: {b}")
                a.rename(b)
            rows.append({"tap": tap, "ten_cu": img.name, "ten_moi_kaggle": f"{moi}{img.suffix}"})
    if not rows:
        print("Không còn tên nào cần đổi.")
        return
    bang = pd.concat([cu, pd.DataFrame(rows)], ignore_index=True)
    bang.to_csv(bang_path, index=False, encoding="utf-8-sig")
    k = pd.read_csv(DATA / "kiem_tra_nhan.csv", encoding="utf-8-sig")
    k["ten_moi"] = k.ten_moi.replace(dict(zip(bang.ten_cu, bang.ten_moi_kaggle)))
    k.to_csv(DATA / "kiem_tra_nhan.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))
    for tap in TAPS:
        print(tap, {s: len(list((DATA / s / tap).iterdir())) for s in ("images", "labels", "masks")})


if __name__ == "__main__":
    main()
