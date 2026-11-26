"""Hướng 3, bước 0: kiểm tra các bộ dữ liệu polyp trước khi gộp.

1. Bộ nào là khung hình cắt từ video (nhiều ảnh gần giống nhau trong cùng bộ), bộ nào là ảnh riêng lẻ.
2. Ảnh của các bộ có trùng hoặc gần trùng với dữ liệu đã huấn luyện (BG20: Kvasir-SEG 1.000 ảnh + 200 ảnh nền) không.
3. Các bộ có trùng nhau không (để biết khi gộp phải bỏ bản lặp).

Đặc trưng giống kiem_tra_trung_polypgen.py trong Test_Mau (chép, không import): md5 tệp và tương quan Pearson
giữa hai ảnh xám thu về 64×64 sau khi cắt viền đen. Ngưỡng: >= 0,95 coi là trùng/gần trùng.

Chỉ ĐỌC từ DeepLearning/Dataset và Test_Mau/01_du_lieu. Ghi bảng vào ../bang/. Không sửa, không chép ảnh.

    python kiem_tra_video_va_trung.py
"""
from __future__ import annotations

import csv
import hashlib
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "bang"
DL = HERE.parents[3]                       # DeepLearning
DS = DL / "Dataset"
BG20 = DL / "Test_Mau" / "01_du_lieu" / "Kvasir_YOLO_SEG_BG20" / "images"
PG = DS / "PolypGen2021_MultiCenterData_v3" / "PolypGen2021_MultiCenterData_v3"
PDB = DS / "PolypDB"
CORR_DUP = 0.95
CORR_GAN = 0.90
EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def read(p: Path):
    return cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)


def crop_black(gray):
    m = gray > 20
    if m.sum() < 100:
        return gray
    ys, xs = np.where(m)
    return gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def feat(p: Path):
    img = read(p)
    g = crop_black(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    v = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    v = (v - v.mean()) / (v.std() + 1e-6)
    return hashlib.md5(p.read_bytes()).hexdigest(), v, img.shape[1], img.shape[0]


def imgs(d: Path):
    return sorted(p for p in d.iterdir() if p.suffix.lower() in EXT)


def collect():
    """(bộ, nhóm con, đường dẫn). Nhóm con: trung tâm/kiểu ảnh/sequence nếu có."""
    rows = []
    for split in ("train", "val"):
        for p in imgs(BG20 / split):
            rows.append(("BG20_train_cu", "nen" if p.name.startswith("bg_") else "kvasir", p))
    meta = pd.read_csv(DS / "CVC-ClinicDB" / "metadata.csv")
    seq = {Path(r).name: int(s) for r, s in zip(meta.png_image_path, meta.sequence_id)}
    for p in imgs(DS / "CVC-ClinicDB" / "PNG" / "Original"):
        rows.append(("CVC-ClinicDB", f"seq{seq.get(p.name, -1)}", p))
    for p in imgs(DS / "CVC-ColonDB_data" / "CVC-ColonDB" / "images"):
        rows.append(("CVC-ColonDB", "", p))
    for p in imgs(DS / "ETIS-Larib PolypDB" / "images"):
        rows.append(("ETIS-Larib", "", p))
    for c in range(1, 7):
        for p in imgs(PG / f"data_C{c}" / f"images_C{c}"):
            rows.append(("PolypGen_don", f"C{c}", p))
    center = {}
    for d in (PDB / "PolypDB_center_wise").glob("*/*/images"):
        for p in imgs(d):
            center[p.name] = f"{d.parent.parent.name}/{d.parent.name}"
    for d in sorted((PDB / "PolypDB_modality_wise").glob("*/images")):
        for p in imgs(d):
            rows.append(("PolypDB", center.get(p.name, d.parent.name), p))
    for sub in ("train/train", "test/test"):
        for p in imgs(DS / "bkai-igh-neopolyp" / sub):
            rows.append(("BKAI-IGH", sub.split("/")[0], p))
    return rows


class UF:
    def __init__(s, n): s.p = list(range(n))
    def f(s, x):
        while s.p[x] != x:
            s.p[x] = s.p[s.p[x]]; x = s.p[x]
        return x
    def u(s, a, b): s.p[s.f(a)] = s.f(b)


def natkey(name):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name)]


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    OUT.mkdir(parents=True, exist_ok=True)
    rows = collect()
    print("Số ảnh mỗi bộ:", dict(Counter(r[0] for r in rows)))
    cache = OUT / "_dac_trung.npz"
    paths = [str(r[2]) for r in rows]
    if cache.exists() and list(np.load(cache, allow_pickle=True)["paths"]) == paths:
        z = np.load(cache, allow_pickle=True); V, md5, wh = z["V"], list(z["md5"]), z["wh"]
    else:
        F = []
        for i, r in enumerate(rows):
            F.append(feat(r[2]))
            if (i + 1) % 1000 == 0:
                print(f"  đã đọc {i + 1}/{len(rows)}", flush=True)
        md5 = [f[0] for f in F]; V = np.stack([f[1] for f in F]); wh = np.array([(f[2], f[3]) for f in F])
        np.savez_compressed(cache, V=V, md5=np.array(md5), wh=wh, paths=np.array(paths))
    ds = np.array([r[0] for r in rows]); grp = [r[1] for r in rows]; names = [r[2].name for r in rows]
    n = len(rows); d = V.shape[1]

    # tương quan theo khối, giữ: láng giềng gần nhất trong bộ, gần nhất với BG20, gần nhất ở bộ khác
    nn_in = np.full(n, -1.0); nn_in_j = np.zeros(n, int)
    nn_bg = np.full(n, -1.0); nn_bg_j = np.zeros(n, int)
    nn_ot = np.full(n, -1.0); nn_ot_j = np.zeros(n, int)
    pairs_in = []      # (i, j, c) trong cùng bộ, c >= CORR_GAN
    pairs_x = []       # khác bộ, c >= CORR_GAN
    isbg = ds == "BG20_train_cu"
    B = 1024
    for a in range(0, n, B):
        C = V[a:a + B] @ V.T / d
        for k in range(C.shape[0]):
            i = a + k
            c = C[k]; c[i] = -2
            same = ds == ds[i]
            j = int(np.argmax(np.where(same, c, -2))); nn_in[i], nn_in_j[i] = c[j], j
            j = int(np.argmax(np.where(isbg, c, -2))); nn_bg[i], nn_bg_j[i] = c[j], j
            j = int(np.argmax(np.where(~same & ~isbg, c, -2))); nn_ot[i], nn_ot_j[i] = c[j], j
            hit = np.where(c >= CORR_GAN)[0]
            for j in hit[hit > i]:
                (pairs_in if ds[j] == ds[i] else pairs_x).append((i, int(j), float(c[j])))
    md5_idx = defaultdict(list)
    for i, m in enumerate(md5):
        md5_idx[m].append(i)

    # 1. từng ảnh
    per = pd.DataFrame({
        "bo": ds, "nhom": grp, "anh": names, "w": wh[:, 0], "h": wh[:, 1],
        "nn_cung_bo": nn_in.round(4), "anh_nn_cung_bo": [names[j] for j in nn_in_j],
        "nn_bg20": nn_bg.round(4), "anh_nn_bg20": [names[j] for j in nn_bg_j],
        "nn_bo_khac": nn_ot.round(4), "bo_nn_khac": [ds[j] for j in nn_ot_j], "anh_nn_bo_khac": [names[j] for j in nn_ot_j],
        "md5_trung_bg20": [any(isbg[j] for j in md5_idx[m] if j != i) for i, m in enumerate(md5)],
        "duong_dan": paths,
    })
    per.to_csv(OUT / "01_tung_anh.csv", index=False, encoding="utf-8-sig")

    # 2. dấu hiệu video theo bộ
    out = []
    for b in dict.fromkeys(ds):
        idx = np.where(ds == b)[0]
        uf = UF(n)
        for i, j, c in pairs_in:
            if ds[i] == b: uf.u(i, j)
        sizes = Counter(uf.f(i) for i in idx)
        order = sorted(idx, key=lambda i: (grp[i], natkey(names[i])))
        cons = [float(V[i] @ V[j] / d) for i, j in zip(order, order[1:]) if grp[i] == grp[j]]
        out.append({
            "bo": b, "so_anh": len(idx),
            "nn_trung_vi": round(float(np.median(nn_in[idx])), 3),
            "ty_le_nn_ge_0_90": round(float((nn_in[idx] >= CORR_GAN).mean()), 3),
            "ty_le_nn_ge_0_95": round(float((nn_in[idx] >= CORR_DUP).mean()), 3),
            "so_cum_0_90": len(sizes), "cum_lon_nhat": max(sizes.values()),
            "anh_trong_cum_ge2": int(sum(s for s in sizes.values() if s >= 2)),
            "ke_nhau_theo_ten_trung_vi": round(float(np.median(cons)), 3) if cons else None,
            "md5_lap_trong_bo": int(sum(len(v) - 1 for v in md5_idx.values() if all(ds[j] == b for j in v) and len(v) > 1)),
        })
    vid = pd.DataFrame(out)
    vid.to_csv(OUT / "02_dau_hieu_video.csv", index=False, encoding="utf-8-sig")

    # 3. trùng với BG20 (dữ liệu huấn luyện cũ)
    t = []
    for b in dict.fromkeys(ds):
        if b == "BG20_train_cu": continue
        m = per[per.bo == b]
        t.append({"bo": b, "so_anh": len(m), "md5_trung_bg20": int(m.md5_trung_bg20.sum()),
                  "corr_ge_0_95_voi_bg20": int((m.nn_bg20 >= CORR_DUP).sum()),
                  "corr_0_90_0_95_voi_bg20": int(((m.nn_bg20 >= CORR_GAN) & (m.nn_bg20 < CORR_DUP)).sum()),
                  "corr_max_voi_bg20": round(float(m.nn_bg20.max()), 3)})
    bg = pd.DataFrame(t)
    bg.to_csv(OUT / "03_trung_voi_BG20.csv", index=False, encoding="utf-8-sig")
    per[(per.bo != "BG20_train_cu") & ((per.nn_bg20 >= CORR_GAN) | per.md5_trung_bg20)] \
        .sort_values("nn_bg20", ascending=False).to_csv(OUT / "03b_cap_trung_BG20.csv", index=False, encoding="utf-8-sig")

    # 4. trùng giữa các bộ (không tính BG20)
    cross = Counter()
    for i, j, c in pairs_x:
        if isbg[i] or isbg[j]: continue
        if c >= CORR_DUP:
            cross[tuple(sorted((f"{ds[i]}|{grp[i].split('/')[0]}", f"{ds[j]}|{grp[j].split('/')[0]}")))] += 1
    md5x = Counter()
    for v in md5_idx.values():
        bs = sorted({f"{ds[j]}|{grp[j].split('/')[0]}" for j in v if not isbg[j]})
        for a in range(len(bs)):
            for b2 in bs[a + 1:]:
                md5x[(bs[a], b2)] += 1
    keys = sorted(set(cross) | set(md5x))
    pd.DataFrame([{"bo_a": k[0], "bo_b": k[1], "cap_corr_ge_0_95": cross[k], "nhom_md5_trung": md5x[k]} for k in keys]) \
        .to_csv(OUT / "04_trung_giua_cac_bo.csv", index=False, encoding="utf-8-sig")

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print("\n== Dấu hiệu video ==\n", vid.to_string(index=False))
        print("\n== Trùng với BG20 ==\n", bg.to_string(index=False))
        print("\n== Trùng giữa các bộ (corr >= 0,95 / md5) ==")
        for k in keys:
            print(f"  {k[0]:28s} ~ {k[1]:28s} corr {cross[k]:5d}   md5 {md5x[k]:5d}")


if __name__ == "__main__":
    main()
