"""Hướng 3, bước 3: tạo bộ dữ liệu YOLO-seg 3 tập từ chon_du_lieu/danh_sach_chon.csv.

Ghi ../du_lieu/H3_Gop_3tap/ (dừng nếu đã có):
  images/{train,val,test}/   ảnh đã thu nhỏ cạnh dài <= 640 (chép từ anh_gom/)
  labels/{train,val,test}/   nhãn polygon YOLO; ảnh âm tính có tệp nhãn rỗng (0 byte), như BG20
  masks/{train,val,test}/    mặt nạ nhị phân PNG 0/255 cùng kích thước ảnh (NEAREST), dùng cho đánh giá theo pixel
  data_h3_3tap.yaml, kiem_tra_nhan.csv
Đổi mặt nạ -> polygon: chép đúng hàm convert_mask_to_yolo_polygons của báo cáo
(Test_Mau/01_du_lieu/scripts_tien_xu_ly/convert_kvasir_with_background_to_yolo_seg.py): Otsu, MORPH_CLOSE 3x3,
RETR_EXTERNAL, bỏ vùng < 20 px, approxPolyDP 0,002 x chu vi. Polygon tính trên mặt nạ GỐC (độ phân giải gốc),
tọa độ chuẩn hóa nên dùng được cho ảnh đã thu nhỏ.
Tên tệp giữ tên trong anh_gom (có tiền tố nguồn). Chỉ đọc dữ liệu gốc.
"""
from __future__ import annotations

import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

H3 = Path(__file__).resolve().parent.parent
GOM = H3 / "anh_gom"
CHON = H3 / "chon_du_lieu"
OUT = H3 / "du_lieu" / "H3_Gop_3tap"
TAPS = ["train", "val", "test"]


def convert_mask_to_yolo_polygons(mask_path: Path, min_area: float = 20.0, epsilon_ratio: float = 0.002):
    """Chép nguyên logic từ script BG20 của báo cáo (bỏ phần thống kê)."""
    mask = cv2.imdecode(np.fromfile(str(mask_path), np.uint8), cv2.IMREAD_COLOR)
    if mask is None:
        raise ValueError(f"Không đọc được mặt nạ: {mask_path}")
    gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) if mask.ndim == 3 else mask
    height, width = gray.shape[:2]
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        pts_orig = cnt.squeeze()
        approx = cv2.approxPolyDP(cnt, epsilon_ratio * cv2.arcLength(cnt, True), True).squeeze()
        if len(approx.shape) < 2 or len(approx) < 3:
            approx = pts_orig
            if len(approx.shape) < 2 or len(approx) < 3:
                continue
        coords = [f"{np.clip(p[0] / width, 0, 1):.6f} {np.clip(p[1] / height, 0, 1):.6f}" for p in approx]
        lines.append(f"0 {' '.join(coords)}\n")
    return lines, (width, height)


def lam_mot(r):
    tap = r.tap
    src_img = GOM / r.thu_muc / r.ten_moi
    shutil.copy2(src_img, OUT / "images" / tap / r.ten_moi)
    with Image.open(src_img) as im:
        w, h = im.size
    stem = Path(r.ten_moi).stem
    if r.thu_muc == "polyp":
        lines, (W0, H0) = convert_mask_to_yolo_polygons(Path(r.duong_dan_mask_goc))
        m = cv2.imdecode(np.fromfile(r.duong_dan_mask_goc, np.uint8), cv2.IMREAD_GRAYSCALE)
        m = (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) * 255
    else:
        lines, m = [], np.zeros((h, w), np.uint8)
    (OUT / "labels" / tap / f"{stem}.txt").write_text("".join(lines), encoding="ascii")
    cv2.imencode(".png", m)[1].tofile(str(OUT / "masks" / tap / f"{stem}.png"))
    # độ khớp polygon với mặt nạ (Dice giữa polygon vẽ lại và mặt nạ đã thu nhỏ)
    poly = np.zeros((h, w), np.uint8)
    for ln in lines:
        xy = np.array(ln.split()[1:], float).reshape(-1, 2) * [w, h]
        cv2.fillPoly(poly, [np.round(xy).astype(np.int32)], 255)
    a, b = poly > 0, m > 0
    dice = 1.0 if a.sum() + b.sum() == 0 else 2 * (a & b).sum() / (a.sum() + b.sum())
    return {"ten_moi": r.ten_moi, "tap": tap, "loai": r.thu_muc, "w": w, "h": h, "so_polygon": len(lines),
            "so_diem": sum(len(ln.split()) // 2 for ln in lines), "dice_polygon_vs_mask": round(float(dice), 4)}


def ve_mau(k: pd.DataFrame):
    font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
    rng = np.random.default_rng(2026)
    rows = []
    for tap in TAPS:
        g = k[(k.tap == tap) & (k.loai == "polyp")]
        rows.append((tap, g.iloc[np.sort(rng.choice(len(g), 10, replace=False))]))
    cao, W = 140, 10 * 200
    can = Image.new("RGB", (W, len(rows) * (cao + 22) + 10), "white")
    d = ImageDraw.Draw(can)
    for ri, (tap, g) in enumerate(rows):
        for ci, r in enumerate(g.itertuples()):
            stem = Path(r.ten_moi).stem
            im = cv2.imdecode(np.fromfile(str(OUT / "images" / tap / r.ten_moi), np.uint8), cv2.IMREAD_COLOR)
            for ln in (OUT / "labels" / tap / f"{stem}.txt").read_text().splitlines():
                xy = np.array(ln.split()[1:], float).reshape(-1, 2) * [r.w, r.h]
                cv2.polylines(im, [np.round(xy).astype(np.int32)], True, (0, 255, 0), 3)
            pil = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            pil.thumbnail((195, cao))
            can.paste(pil, (ci * 200, ri * (cao + 22)))
            d.text((ci * 200, ri * (cao + 22) + cao + 3), f"{tap} {r.ten_moi[:20]}", fill="black", font=font)
    p = CHON / "hinh" / "h08_nhan_yolo_mau_tung_tap.jpg"
    can.save(p, quality=88)
    print("hình:", p.name)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if OUT.exists():
        sys.exit(f"{OUT} đã có, dừng để không ghi đè.")
    c = pd.read_csv(CHON / "danh_sach_chon.csv", encoding="utf-8-sig").fillna({"duong_dan_mask_goc": ""})
    for sub in ("images", "labels", "masks"):
        for t in TAPS:
            (OUT / sub / t).mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(8) as ex:
        k = pd.DataFrame(list(ex.map(lam_mot, c.itertuples())))
    k.to_csv(OUT / "kiem_tra_nhan.csv", index=False, encoding="utf-8-sig")
    cnt = k.pivot_table(index="tap", columns="loai", values="ten_moi", aggfunc="count").reindex(TAPS)
    (OUT / "data_h3_3tap.yaml").write_text(f"""# Hướng 3: dữ liệu gộp chia 3 tập (Kvasir-SEG, PolypGen, PolypDB Simula WLI/NBI, Karolinska + ảnh âm tính)
# Tạo bằng Huong3_GopDuLieu/scripts/dong_goi_3tap.py từ chon_du_lieu/danh_sach_chon.csv. Trên Kaggle sửa 'path'.
path: .
train: images/train
val: images/val
test: images/test

names:
  0: polyp

metadata:
  train: {int(cnt.loc['train'].sum())}  # {int(cnt.loc['train','polyp'])} polyp + {int(cnt.loc['train','am_tinh'])} âm tính
  val: {int(cnt.loc['val'].sum())}  # {int(cnt.loc['val','polyp'])} polyp + {int(cnt.loc['val','am_tinh'])} âm tính
  test: {int(cnt.loc['test'].sum())}  # {int(cnt.loc['test','polyp'])} polyp + {int(cnt.loc['test','am_tinh'])} âm tính
  split_seed: 2026
  image_max_side: 640
""", encoding="utf-8")
    print(cnt.to_string())
    pol, neg = k[k.loai == "polyp"], k[k.loai == "am_tinh"]
    print("Polyp không có polygon:", int((pol.so_polygon == 0).sum()), "| âm tính có polygon:", int((neg.so_polygon > 0).sum()))
    print("Số polygon/ảnh polyp:", pol.so_polygon.value_counts().sort_index().to_dict())
    print("Dice polygon vs mặt nạ (polyp): trung vị", pol.dice_polygon_vs_mask.median(), "| min", pol.dice_polygon_vs_mask.min(),
          "| < 0,90:", int((pol.dice_polygon_vs_mask < 0.90).sum()))
    print("Cạnh dài ảnh: max", int(k[["w", "h"]].max(axis=1).max()))
    for t in TAPS:
        n = {s: len(list((OUT / s / t).iterdir())) for s in ("images", "labels", "masks")}
        print(t, n)
    ve_mau(k)


if __name__ == "__main__":
    main()
