"""Đối chiếu kết quả Krokiet (Czkawka 12.0.2, Similar Images, người dùng tự quét anh_gom/) với bộ chọn và chia của script.

Đọc anh_gom/anh_trung/results_similar_images_compact.json, ghi chon_du_lieu/doi_chieu_krokiet.csv (từng nhóm) và
hình chon_du_lieu/hinh/h07_krokiet_*.jpg. Chỉ đọc, không sửa ảnh.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, ImageDraw, ImageFont

H3 = Path(__file__).resolve().parent.parent
GOM = H3 / "anh_gom"
OUT = H3 / "chon_du_lieu"
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 12)
FONT_TO = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 20)


def nguon(ten: str) -> str:
    t = ten.split("_")[0]
    return {"PG": "PG_" + ten.split("_")[1], "PGR": "PGR", "PGS": "PGS_" + ten.split("_")[1]}.get(t, t)


def ve(nhom, tieu_de, path, cao=110):
    if not nhom:
        return
    rows = []
    for g, items in nhom:
        rows.append((g, items))
    W = 8 * (cao * 16 // 9 + 6) + 120
    H = 36 + len(rows) * (cao + 20)
    can = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(can)
    d.text((8, 6), tieu_de, fill="black", font=FONT_TO)
    y = 36
    for g, items in rows:
        d.text((4, y + cao // 2), f"#{g}", fill="black", font=FONT)
        x = 50
        for ten, thu_muc, nhan in items[:8]:
            im = Image.open(GOM / thu_muc / ten).convert("RGB")
            im.thumbnail((cao * 16 // 9, cao))
            can.paste(im, (x, y))
            d.text((x, y + cao + 2), nhan, fill="black", font=FONT)
            x += im.width + 6
        y += cao + 20
    can.save(path, quality=85)
    print("hình:", path.name, can.size)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    js = json.loads((GOM / "anh_trung" / "results_similar_images_compact.json").read_text(encoding="utf-8"))
    c = pd.read_csv(OUT / "danh_sach_chon.csv", encoding="utf-8-sig")
    tap, cum = dict(zip(c.ten_moi, c.tap)), dict(zip(c.ten_moi, c.cum))
    rows, ve_ro, ve_khac_nguon, ve_ngoai_cum = [], [], [], []
    for g, grp in enumerate(js):
        ten = [Path(x["path"]).name for x in grp]
        tm = [Path(x["path"]).parent.name for x in grp]
        ng = sorted({nguon(t) for t in ten})
        chon = [t for t in ten if t in tap]
        taps = sorted({tap[t] for t in chon})
        cums = {cum[t] for t in chon}
        loai = ("chuoi_video_cung_chuoi" if len(ng) == 1 and ng[0].startswith("PGS") else
                "khac_nguon" if len({n.split('_')[0] if not n.startswith('PG') else 'PG' for n in ng}) > 1 else "cung_nguon")
        r = {"nhom": g, "so_anh": len(ten), "nguon": "+".join(ng), "loai": loai, "so_anh_da_chon": len(chon),
             "cac_tap": "+".join(taps), "vat_qua_tap": len(taps) > 1, "chon_khac_cum": len(cums) > 1,
             "max_difference": max(x["difference"] for x in grp), "anh": " | ".join(ten)}
        rows.append(r)
        items = [(t, m, f"{(tap.get(t) or '-')[:2]} {t[:18]}") for t, m in zip(ten, tm)]
        if r["vat_qua_tap"]:
            ve_ro.append((g, items))
        elif loai == "khac_nguon":
            ve_khac_nguon.append((g, items))
        elif r["chon_khac_cum"]:
            ve_ngoai_cum.append((g, items))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "doi_chieu_krokiet.csv", index=False, encoding="utf-8-sig")
    print("Số nhóm:", len(df), "| ảnh:", int(df.so_anh.sum()))
    print(df.groupby("loai").agg(nhom=("nhom", "size"), anh=("so_anh", "sum")).to_string())
    print("Nhóm có >= 2 ảnh đã chọn:", int((df.so_anh_da_chon >= 2).sum()))
    print("Nhóm vắt qua 2 tập (rò rỉ):", int(df.vat_qua_tap.sum()))
    print("Nhóm cùng tập nhưng script để khác cụm:", int((df.chon_khac_cum & ~df.vat_qua_tap).sum()))
    print("Nhóm khác nguồn:", int((df.loai == "khac_nguon").sum()))
    print(df[df.loai == "khac_nguon"].nguon.value_counts().to_string())
    ve(ve_ro, f"Krokiet: {len(ve_ro)} nhóm có ảnh đã chọn nằm ở 2 tập khác nhau", OUT / "hinh" / "h07a_krokiet_vat_qua_tap.jpg")
    ve(ve_khac_nguon, f"Krokiet: {len(ve_khac_nguon)} nhóm giữa hai nguồn khác nhau (tr/va/te = tập, - = không chọn)",
       OUT / "hinh" / "h07b_krokiet_khac_nguon.jpg")


if __name__ == "__main__":
    main()
