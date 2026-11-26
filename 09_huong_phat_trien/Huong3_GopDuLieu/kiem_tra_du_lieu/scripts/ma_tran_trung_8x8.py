"""Hướng 3, bước 0b: ma trận trùng ảnh 8×8, ghi lại MỌI cặp có tương quan ≥ 0,85.

8 bộ: BG20 (Kvasir-SEG 1.000 + 200 ảnh nền đã huấn luyện), PolypGen ảnh đơn, PolypDB Simula WLI,
PolypDB Simula NBI, PolypDB Karolinska (5 bộ dùng để gộp), và CVC-ClinicDB, CVC-ColonDB, ETIS-Larib (3 bộ test ngoài).

Dùng lại đặc trưng đã lưu trong ../bang/_dac_trung.npz (ảnh xám 64×64 sau cắt viền đen, đã chuẩn hóa), do
kiem_tra_video_va_trung.py tạo. Thêm phép so với ảnh lật ngang, lật dọc và xoay 180° để bắt bản sao bị lật.
Tương quan của một cặp = max qua 4 phép (giữ nguyên, lật ngang, lật dọc, xoay 180°).

Ghi ../bang/07_cap_ge_0_85.csv (mọi cặp, kể cả trong cùng bộ) và 08_ma_tran_8x8.csv. Chỉ đọc ảnh, không sửa gì.
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
BANG = HERE.parent / "bang"
NGUONG = 0.85
BO = ["BG20", "PolypGen", "Simula_WLI", "Simula_NBI", "Karolinska", "CVC-ClinicDB", "CVC-ColonDB", "ETIS-Larib"]


def ten_bo(bo, nhom):
    if bo == "BG20_train_cu": return "BG20"
    if bo == "PolypGen_don": return "PolypGen"
    if bo == "PolypDB":
        return {"Simula/WLI": "Simula_WLI", "Simula/NBI": "Simula_NBI", "Karolinska/WLI": "Karolinska"}.get(nhom)
    return bo if bo in BO else None


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = pd.read_csv(BANG / "01_tung_anh.csv", encoding="utf-8-sig")
    z = np.load(BANG / "_dac_trung.npz", allow_pickle=True)
    assert list(z["paths"]) == list(p.duong_dan), "đặc trưng không khớp 01_tung_anh.csv"
    p["set"] = [ten_bo(b, str(n)) for b, n in zip(p.bo, p.nhom)]
    keep = p.set.notna().to_numpy()
    p = p[keep].reset_index(drop=True)
    V = z["V"][keep].astype(np.float32)
    md5 = z["md5"][keep]
    n, d = V.shape
    print("Số ảnh:", p.set.value_counts().reindex(BO).to_dict(), "tổng", n)
    img = V.reshape(n, 64, 64)
    T = {"giu_nguyen": V, "lat_ngang": img[:, :, ::-1].reshape(n, d), "lat_doc": img[:, ::-1, :].reshape(n, d),
         "xoay_180": img[:, ::-1, ::-1].reshape(n, d)}
    sets = p.set.to_numpy()
    rows = []
    B = 512
    for a in range(0, n, B):
        best = np.full((min(B, n - a), n), -2.0, np.float32)
        how = np.zeros_like(best, dtype=np.int8)
        for k, (name, W) in enumerate(T.items()):
            c = V[a:a + B] @ W.T / d
            m = c > best
            best[m], how[m] = c[m], k
        for i_loc in range(best.shape[0]):
            i = a + i_loc
            js = np.where(best[i_loc] >= NGUONG)[0]
            for j in js[js > i]:
                rows.append((i, int(j), float(best[i_loc, j]), list(T)[how[i_loc, j]]))
    for i in range(n):           # md5 trùng dù tương quan thấp (không xảy ra, nhưng kiểm cho chắc)
        pass
    md5_pairs = {(i, j) for v in pd.Series(range(n)).groupby(md5).groups.values() if len(v) > 1
                 for x, i in enumerate(sorted(v)) for j in sorted(v)[x + 1:]}
    df = pd.DataFrame(rows, columns=["i", "j", "corr", "phep"])
    extra = [(i, j) for i, j in md5_pairs if not ((df.i == i) & (df.j == j)).any()]
    if extra:
        df = pd.concat([df, pd.DataFrame([(i, j, np.nan, "md5") for i, j in extra], columns=df.columns)])
    df["md5_trung"] = [(i, j) in md5_pairs for i, j in zip(df.i, df.j)]
    df["bo_a"], df["bo_b"] = sets[df.i], sets[df.j]
    df["anh_a"], df["anh_b"] = p.anh.to_numpy()[df.i], p.anh.to_numpy()[df.j]
    df["path_a"], df["path_b"] = p.duong_dan.to_numpy()[df.i], p.duong_dan.to_numpy()[df.j]
    # sắp bo_a/bo_b theo thứ tự BO
    sw = [BO.index(x) > BO.index(y) for x, y in zip(df.bo_a, df.bo_b)]
    for c1, c2 in [("bo_a", "bo_b"), ("anh_a", "anh_b"), ("path_a", "path_b")]:
        df.loc[sw, [c1, c2]] = df.loc[sw, [c2, c1]].to_numpy()
    df["dai"] = pd.cut(df["corr"], [NGUONG, 0.90, 0.95, 1.01], right=False, labels=["0,85–0,90", "0,90–0,95", "≥0,95"])
    df = df.sort_values(["bo_a", "bo_b", "corr"], ascending=[True, True, False])
    df.drop(columns=["i", "j"]).to_csv(BANG / "07_cap_ge_0_85.csv", index=False, encoding="utf-8-sig")

    mt = []
    for x in range(len(BO)):
        for y in range(x, len(BO)):
            g = df[(df.bo_a == BO[x]) & (df.bo_b == BO[y])]
            mt.append({"bo_a": BO[x], "bo_b": BO[y], "cung_bo": x == y, "md5": int(g.md5_trung.sum()),
                       "ge_0_95": int((g["corr"] >= 0.95).sum()), "0_90_0_95": int(((g["corr"] >= 0.90) & (g["corr"] < 0.95)).sum()),
                       "0_85_0_90": int((g["corr"] < 0.90).sum()),
                       "nho_lat_xoay_ge_0_90": int(((g["corr"] >= 0.90) & (g.phep != "giu_nguyen")).sum())})
    mt = pd.DataFrame(mt)
    mt.to_csv(BANG / "08_ma_tran_8x8.csv", index=False, encoding="utf-8-sig")
    with pd.option_context("display.width", 200, "display.max_rows", 100):
        print(mt.to_string(index=False))
    print("Tổng số cặp ≥ 0,85:", len(df), "| theo phép:", Counter(df.phep))


if __name__ == "__main__":
    main()
