"""Hướng 3, bước 2: chọn 3.000 ảnh polyp + ảnh âm tính đa dạng (khoảng 14–15%), kiểm tra mặt nạ, gom cụm, chia 70/15/15.

Chỉ đọc ảnh/mặt nạ gốc và anh_gom/, ghi bảng và hình xem trước vào ../chon_du_lieu/. Chưa đóng gói dataset.

Chọn ảnh polyp (3.000):
- giữ hết: Kvasir-SEG 1.000 (để nối với hướng 1, 2), PolypGen ảnh đơn 1.412 (6 trung tâm), Simula NBI 146, Karolinska 30;
- bỏ mọi ảnh polyp có mặt nạ < 0,2% hoặc > 70% diện tích ảnh (người dùng duyệt 19/11), danh sách ở anh_bo_mat_na_bat_thuong.csv;
- Simula WLI: lấy ngẫu nhiên phần còn thiếu để đủ 3.000 (seed 2026), mỗi cụm ảnh gần giống (tương quan >= 0,90 trong bộ) chỉ lấy tối đa 1 ảnh,
  bỏ 5 ảnh WLI trùng tên với ảnh NBI (cùng tổn thương chụp hai chế độ, NBI đã được giữ).
Chọn ảnh âm tính (509):
- 125 ảnh PolypGen ảnh đơn có mặt nạ rỗng (nhiều trung tâm, có nhuộm màu, phân);
- 23 chuỗi PolypGen âm tính x 8 khung cách đều theo số thứ tự khung = 184;
- 200 ảnh Kvasir normal-cecum đã dùng trong BG20.
Chia: gom cụm bằng union-find (tương quan >= 0,90 giữa mọi ảnh đã chọn, md5 trùng, cặp trùng tên WLI/NBI, cùng một chuỗi
âm tính), rồi chia theo từng nguồn, cả cụm vào cùng một tập, mục tiêu 70/15/15.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).resolve().parent
H3 = HERE.parent
DL = H3.parent.parent
GOM = H3 / "anh_gom"
BANG0 = H3 / "kiem_tra_du_lieu" / "bang"
OUT = H3 / "chon_du_lieu"
HINH = OUT / "hinh"
SEED = 2026
N_POLYP = 3000                       # Simula WLI lấy ngẫu nhiên phần còn thiếu
MIN_POLYP, MAX_POLYP = 0.002, 0.70   # tỉ lệ diện tích polyp trên ảnh được giữ
KHUNG_MOI_CHUOI = 8
TI_LE = {"train": 0.70, "val": 0.15, "test": 0.15}
KV_MASK = DL / "Dataset" / "Kvasir-SEG_data" / "Kvasir-SEG" / "Kvasir-SEG" / "masks"
PG = DL / "Dataset" / "PolypGen2021_MultiCenterData_v3" / "PolypGen2021_MultiCenterData_v3"
FONT = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 13)
FONT_TO = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 22)


def nguon_gon(t: str) -> str:
    return "PGS" if t.startswith("PGS") else ("PGR" if t.startswith("PGR") else t)


def duong_dan_mask(r) -> Path | None:
    f = Path(r.duong_dan_goc)
    if r.tien_to == "KV":
        return KV_MASK / f.name
    if r.tien_to.startswith(("PG_", "PGR_")):
        c = r.nhom
        s = f.stem.rstrip("]")
        return PG / f"data_{c}" / f"masks_{c}" / f"{s}_mask.jpg"
    if r.tien_to in ("SW", "SN", "KA"):
        return f.parent.parent / "masks" / (f.stem + ".png")
    return None


class UF:
    def __init__(self, items):
        self.p = {x: x for x in items}

    def f(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def u(self, a, b):
        a, b = self.f(a), self.f(b)
        if a != b:
            self.p[b] = a


def cap_090() -> pd.DataFrame:
    c = pd.read_csv(BANG0 / "07_cap_ge_0_85.csv", encoding="utf-8-sig", usecols=["corr", "path_a", "path_b"])
    return c[c["corr"] >= 0.90]


def chon(d: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    d = d.copy()
    d["nguon_gon"] = d.tien_to.map(nguon_gon)
    # Người dùng duyệt 19/11: bỏ ảnh polyp có mặt nạ < 0,2% ảnh (polyp chỉ là một chấm) hoặc > 70% ảnh
    # (tổn thương lan khắp thành ruột, không còn là một khối riêng). Áp cho mọi nguồn polyp; Simula WLI bù vào để đủ 3.000.
    bat_thuong = (d.thu_muc == "polyp") & ((d.ti_le_polyp < MIN_POLYP) | (d.ti_le_polyp > MAX_POLYP))
    d[bat_thuong].to_csv(OUT / "anh_bo_mat_na_bat_thuong.csv", index=False, encoding="utf-8-sig")
    print("Bỏ vì mặt nạ bất thường:", d[bat_thuong].tien_to.map(lambda t: t[:2] if t.startswith("PG") else t).value_counts().to_dict())
    d = d[~bat_thuong]
    giu = d.tien_to.isin(["KV", "SN", "KA"]) | d.tien_to.str.startswith("PG_") | d.tien_to.str.startswith("PGR_")
    giu |= (d.tien_to == "CE") & d.da_dung_BG20
    # PolypGen chuỗi âm tính: 8 khung cách đều theo số thứ tự
    pgs = d[d.nguon_gon == "PGS"].copy()
    pgs["so"] = pgs.duong_dan_goc.str.extract(r"_(\d+)\.jpg$")[0].astype(int)
    for _, g in pgs.groupby("tien_to"):
        g = g.sort_values("so")
        idx = np.unique(np.round(np.linspace(0, len(g) - 1, KHUNG_MOI_CHUOI)).astype(int))
        giu[g.index[idx]] = True
    # Simula WLI
    sn_ten = {Path(x).name for x in d[d.tien_to == "SN"].duong_dan_goc}
    sw = d[(d.tien_to == "SW") & ~d.duong_dan_goc.map(lambda x: Path(x).name in sn_ten)]
    sw_set = set(sw.duong_dan_goc)
    uf = UF(sw_set)
    for a, b in zip(cap.path_a, cap.path_b):
        if a in sw_set and b in sw_set:
            uf.u(a, b)
    da_lay = set(d[giu].duong_dan_goc)
    ke = {}                                   # ảnh SW gần giống (>= 0,90) với ảnh đã giữ thì không lấy
    for a, b in zip(cap.path_a, cap.path_b):
        if a in sw_set and b in da_lay:
            ke[a] = 1
        if b in sw_set and a in da_lay:
            ke[b] = 1
    # Khớp điểm ORB giữa Simula WLI và các ảnh polyp đã giữ: bắt bản sao mà tương quan 64×64 bỏ sót
    # (vd. cùng ảnh Kvasir nhưng bản Simula có thêm hộp xanh, cắt khác). Ảnh WLI khớp thì không lấy.
    from khop_diem_dac_trung import cap_cung_canh, la_cung_canh
    pol = d[giu & (d.thu_muc == "polyp")]
    ds = list(pol.duong_dan_goc) + list(sw.duong_dan_goc)
    bo = list(pol.tien_to.str[:2]) + ["SW"] * len(sw)
    kd = cap_cung_canh(ds, [""] * len(ds), [np.nan] * len(ds), nguong_corr=0.65, nhom_bo=bo)
    kd = kd[la_cung_canh(kd["corr"], kd.inlier)].copy()
    kd["anh_a"], kd["anh_b"] = [Path(ds[i]).name for i in kd.i], [Path(ds[j]).name for j in kd.j]
    kd["bo_a"], kd["bo_b"] = [bo[i] for i in kd.i], [bo[j] for j in kd.j]
    kd.drop(columns=["i", "j"]).to_csv(OUT / "trung_khac_bo_khop_diem.csv", index=False, encoding="utf-8-sig")
    for i, j in zip(kd.i, kd.j):
        for x in (ds[i], ds[j]):
            if x in sw_set:
                ke[x] = 1
    print("Khớp điểm khác bộ (polyp đã giữ + WLI):", kd.groupby(["bo_a", "bo_b"]).size().to_dict())
    cum = {}
    for p in sw.duong_dan_goc.sample(frac=1, random_state=SEED):
        r = uf.f(p)
        if r not in cum and p not in ke:
            cum[r] = p
    reps = list(cum.values())
    n_sw = N_POLYP - int((giu & (d.thu_muc == "polyp")).sum())
    pick = rng.choice(len(reps), n_sw, replace=False)
    giu[d.duong_dan_goc.isin([reps[i] for i in pick])] = True
    print(f"Simula WLI: {len(sw)} ứng viên, {len(ke)} gần giống ảnh đã giữ, {len(reps)} đại diện cụm, lấy {n_sw}")
    return d[giu].copy()


def kiem_mask(c: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in c.itertuples():
        mp = duong_dan_mask(r)
        rec = {"duong_dan_mask_goc": str(mp) if mp else "", "mask_ok": True, "ti_le_polyp": 0.0, "so_vung": 0, "ghi_chu_mask": ""}
        if mp is not None:
            if not mp.exists():
                rec.update(mask_ok=False, ghi_chu_mask="không có mặt nạ")
            else:
                m = np.asarray(Image.open(mp).convert("L")) > 127
                if (m.shape[1], m.shape[0]) != (r.w_goc, r.h_goc):
                    rec.update(mask_ok=False, ghi_chu_mask=f"lệch kích thước {m.shape[::-1]} vs ({r.w_goc},{r.h_goc})")
                rec["ti_le_polyp"] = float(m.mean())
                from scipy.ndimage import label
                rec["so_vung"] = int(label(m)[1])
                if r.thu_muc == "polyp" and m.sum() == 0:
                    rec.update(mask_ok=False, ghi_chu_mask="ảnh polyp nhưng mặt nạ rỗng")
                if r.thu_muc == "am_tinh" and m.sum() > 0:
                    rec.update(mask_ok=False, ghi_chu_mask="ảnh âm tính nhưng mặt nạ có polyp")
        rows.append(rec)
    return pd.concat([c.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def chia(c: pd.DataFrame, cap: pd.DataFrame) -> pd.DataFrame:
    c = c.copy()
    ps = list(c.duong_dan_goc)
    s = set(ps)
    uf = UF(ps)
    n_cap = 0
    for a, b in zip(cap.path_a, cap.path_b):
        if a in s and b in s:
            uf.u(a, b); n_cap += 1
    for _, g in c.groupby("md5_goc"):
        for x in g.duong_dan_goc.iloc[1:]:
            uf.u(g.duong_dan_goc.iloc[0], x)
    ten = c.duong_dan_goc.map(lambda x: Path(x).name)
    simula = c[c.tien_to.isin(["SW", "SN"])]
    for _, g in simula.groupby(ten[simula.index]):
        for x in g.duong_dan_goc.iloc[1:]:
            uf.u(g.duong_dan_goc.iloc[0], x)
    for _, g in c[c.nguon_gon == "PGS"].groupby("tien_to"):
        for x in g.duong_dan_goc.iloc[1:]:
            uf.u(g.duong_dan_goc.iloc[0], x)
    # Cặp cùng cảnh theo khớp điểm ORB (bắt khung liền nhau của cùng video mà tương quan < 0,90, vd. C2 001160/001161)
    from khop_diem_dac_trung import cap_cung_canh, la_cung_canh
    la_pg = c.tien_to.str.startswith(("PG_", "PGR_"))
    nhom_ten = np.where(la_pg, "PG_" + c.nhom.astype(str), "").tolist()
    so = c.duong_dan_goc.str.extract(r"(\d+)\]?\.\w+$")[0].astype(float).where(la_pg).tolist()
    kd = cap_cung_canh(ps, nhom_ten, so)
    kd = kd[la_cung_canh(kd["corr"], kd.inlier)]
    for i, j in zip(kd.i, kd.j):
        uf.u(ps[i], ps[j])
    kd = kd.assign(anh_a=c.ten_moi.to_numpy()[kd.i], anh_b=c.ten_moi.to_numpy()[kd.j])
    kd.drop(columns=["i", "j"]).to_csv(OUT / "cap_cung_canh_khop_diem.csv", index=False, encoding="utf-8-sig")
    c["cum"] = c.duong_dan_goc.map(uf.f)
    ma = {k: i for i, k in enumerate(c.cum.unique())}
    c["cum"] = c.cum.map(ma)
    print(f"Cặp >= 0,90 giữa các ảnh đã chọn: {n_cap}; số cụm: {c.cum.nunique()}; cụm >= 2 ảnh: {(c.cum.value_counts() >= 2).sum()}")
    # tầng chia: nguồn (PolypGen theo trung tâm). Cụm nằm ở nhiều tầng thì xếp theo tầng của ảnh đầu tiên.
    c["tang"] = np.where(c.tien_to.str.startswith("PG_"), c.tien_to, c.nguon_gon)
    tang_cum = c.groupby("cum").tang.first()
    rng = np.random.default_rng(SEED)
    tap = {}
    for t, cums in tang_cum.groupby(tang_cum):
        size = c.cum.value_counts()[cums.index]
        order = list(cums.index)
        rng.shuffle(order)
        order.sort(key=lambda k: -size[k])     # cụm lớn trước, cùng cỡ thì theo thứ tự ngẫu nhiên
        dem = dict.fromkeys(TI_LE, 0)
        tong = size.sum()
        for k in order:
            sp = max(TI_LE, key=lambda x: TI_LE[x] * tong - dem[x])
            tap[k] = sp
            dem[sp] += size[k]
    c["tap"] = c.cum.map(tap)
    return c


# ---------- hình xem trước ----------
def thumb(r, cao=150, vien=True) -> Image.Image:
    im = Image.open(GOM / r.thu_muc / r.ten_moi).convert("RGB")
    s = cao / im.height
    im = im.resize((max(1, round(im.width * s)), cao), Image.LANCZOS)
    if vien and r.duong_dan_mask_goc and r.thu_muc == "polyp" and Path(r.duong_dan_mask_goc).exists():
        m = Image.open(r.duong_dan_mask_goc).convert("L").resize(im.size, Image.NEAREST).point(lambda v: 255 if v > 127 else 0)
        er = m.filter(ImageFilter.MinFilter(3))
        edge = Image.fromarray(((np.asarray(m) > 0) & ~(np.asarray(er) > 0)).astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))
        im.paste((0, 255, 0), mask=edge)
    return im


def luoi(items, tieu_de, path, cot=8, cao=150, nhan=None):
    th = [thumb(r, cao) for r in items]
    w = max(t.width for t in th)
    hang = (len(th) + cot - 1) // cot
    W, H = cot * (w + 6) + 6, 40 + hang * (cao + 22) + 6
    can = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(can)
    dr.text((8, 8), tieu_de, fill="black", font=FONT_TO)
    for i, (t, r) in enumerate(zip(th, items)):
        x, y = 6 + (i % cot) * (w + 6), 40 + (i // cot) * (cao + 22)
        can.paste(t, (x + (w - t.width) // 2, y))
        lab = nhan(r) if nhan else r.ten_moi[:24]
        dr.text((x + 2, y + cao + 3), lab, fill="black", font=FONT)
    can.save(path, quality=88)
    print("hình:", path.name, can.size)


def ve(c: pd.DataFrame):
    HINH.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    lay = lambda g, n: [r for r in g.iloc[np.sort(rng.choice(len(g), min(n, len(g)), replace=False))].itertuples()]
    # 1. ảnh polyp: 8 ảnh mỗi nguồn (PolypGen theo trung tâm), viền xanh là mặt nạ
    pol = c[c.thu_muc == "polyp"]
    items = []
    for t in ["KV", "PG_C1", "PG_C2", "PG_C3", "PG_C4", "PG_C5", "PG_C6", "SW", "SN", "KA"]:
        items += lay(pol[pol.tang == t], 8)
    luoi(items, "Ảnh polyp đã chọn: 8 ảnh ngẫu nhiên mỗi nguồn (viền xanh = mặt nạ)", HINH / "h01_polyp_moi_nguon.jpg",
         nhan=lambda r: f"{r.tang} | {r.tap}")
    # 2. 184 khung chuỗi âm tính: mỗi hàng một chuỗi
    pgs = c[c.nguon_gon == "PGS"].copy()
    pgs["so"] = pgs.duong_dan_goc.str.extract(r"_(\d+)\.jpg$")[0].astype(int)
    pgs["seq"] = pgs.tien_to.str.extract(r"seq(\d+)")[0].astype(int)
    pgs = pgs.sort_values(["seq", "so"])
    luoi(list(pgs.itertuples()), "Âm tính: 23 chuỗi PolypGen x 8 khung cách đều (mỗi hàng một chuỗi)", HINH / "h02_am_tinh_chuoi_polypgen.jpg",
         cot=8, cao=110, nhan=lambda r: f"seq{r.seq} #{r.so} | {r.tap}")
    # 3. 125 ảnh mặt nạ rỗng
    pgr = c[c.nguon_gon == "PGR"].sort_values("ten_moi")
    luoi(list(pgr.itertuples()), "Âm tính: 125 ảnh đơn PolypGen có mặt nạ rỗng (cần soi: có polyp bị bỏ sót nhãn không?)",
         HINH / "h03_am_tinh_mat_na_rong.jpg", cot=10, cao=110, nhan=lambda r: f"{r.ten_moi[4:6]} {r.ten_moi[-16:-4]} {r.tap[:2]}")
    # 4. 40 ảnh cecum
    luoi(lay(c[c.nguon_gon == "CE"], 40), "Âm tính: 40/200 ảnh Kvasir normal-cecum", HINH / "h04_am_tinh_cecum.jpg", cot=10, cao=110,
         nhan=lambda r: r.tap)
    # 5. mặt nạ đáng ngờ
    bad = pd.read_csv(OUT / "anh_bo_mat_na_bat_thuong.csv", encoding="utf-8-sig").fillna({"duong_dan_mask_goc": ""})
    bad = bad.sort_values("ti_le_polyp")
    luoi(list(bad.itertuples()), f"Đã bỏ {len(bad)} ảnh polyp: mặt nạ < 0,2% hoặc > 70% ảnh (trên toàn bộ 5.121 ảnh polyp gom)",
         HINH / "h05_anh_da_bo_mat_na_bat_thuong.jpg", cot=8, nhan=lambda r: f"{r.tien_to} {r.ti_le_polyp:.3%}")
    assert c[c.thu_muc == "polyp"].ti_le_polyp.between(MIN_POLYP, MAX_POLYP).all()


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if (OUT / "danh_sach_chon.csv").exists():
        sys.exit(f"{OUT / 'danh_sach_chon.csv'} đã có, dừng để không ghi đè.")
    OUT.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(GOM / "danh_sach_anh.csv", encoding="utf-8-sig")
    cap = cap_090()
    d = kiem_mask(d)                 # mọi ảnh gom; ảnh không có mặt nạ (cecum, chuỗi âm tính) bỏ qua
    print("Mặt nạ lỗi trong toàn bộ ảnh gom:", int((~d.mask_ok).sum()))
    c = chon(d, cap)
    c = chia(c, cap)
    c.drop(columns=["da_dung_BG20"]).to_csv(OUT / "danh_sach_chon.csv", index=False, encoding="utf-8-sig")
    tt = c.pivot_table(index=["thu_muc", "tang"], columns="tap", values="ten_moi", aggfunc="count", fill_value=0)
    tt = tt[list(TI_LE)]
    tt["tong"] = tt.sum(axis=1)
    tt.loc[("TỔNG", ""), :] = tt.sum()
    tt.to_csv(OUT / "tom_tat_chia.csv", encoding="utf-8-sig")
    print(tt.to_string())
    n_am = (c.thu_muc == "am_tinh").sum()
    print(f"Âm tính {n_am}/{len(c)} = {n_am / len(c):.1%}; mặt nạ lỗi: {(~c.mask_ok).sum()}")
    print(c[~c.mask_ok][["ten_moi", "ghi_chu_mask"]].to_string())
    ve(c)


if __name__ == "__main__":
    main()
