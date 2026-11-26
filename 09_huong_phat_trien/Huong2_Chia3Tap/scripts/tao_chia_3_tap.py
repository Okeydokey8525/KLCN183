"""Hướng 2: chia lại BG20 (Kvasir-SEG + 20% ảnh nền cecum) thành 3 tập train / val / test.

Quy tắc (chốt 17/11/2026, trước khi huấn luyện):
- TEST = nguyên tập val cũ (120 polyp + 40 nền). 120 ảnh polyp này thuộc split 880/120 do tác giả Kvasir-SEG
  cung cấp. Giữ nguyên để so được với Bảng 4.8 trên cùng ảnh, chỉ khác là chọn checkpoint không còn nhìn thấy test.
- VAL mới = 120 polyp + 40 nền tách từ train cũ (cùng cỡ với test). TRAIN mới = 760 polyp + 120 nền còn lại.
- Ảnh polyp chọn vào val theo phân tầng 3 nhóm diện tích polyp (tertile tính trên 880 ảnh train cũ).
- Ảnh gần trùng trong train cũ (tương quan ảnh xám 64×64 sau cắt viền đen >= 0,95, giống
  Test_Mau/08_du_lieu_thuc_te/scripts/kiem_tra_trung_polypgen.py) được gom thành cụm; cả cụm vào cùng một tập.
- Một seed chia cố định: SEED_CHIA = 2026. Không chia lại sau khi đã thấy kết quả.

Chỉ ĐỌC Test_Mau/01_du_lieu/Kvasir_YOLO_SEG_BG20. Ghi vào Huong2_Chia3Tap/du_lieu/Kvasir_BG20_3tap (chép file,
không dùng hard link) và Huong2_Chia3Tap/du_lieu/*.csv|json.

    C:\\venv_kl_gpu\\Scripts\\python.exe tao_chia_3_tap.py [--chi-kiem-tra]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
H2 = HERE.parent                                                   # Huong2_Chia3Tap
TEST_MAU = H2.parent.parent / "Test_Mau"
SRC = TEST_MAU / "01_du_lieu" / "Kvasir_YOLO_SEG_BG20"
OUT_ROOT = H2 / "du_lieu"
OUT = OUT_ROOT / "Kvasir_BG20_3tap"
SEED_CHIA = 2026
N_VAL_POLYP, N_VAL_NEN = 120, 40
CORR_DUP = 0.95


def crop_black(gray):
    m = gray > 20
    if m.sum() < 100:
        return gray
    ys, xs = np.where(m)
    return gray[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def dac_trung(p):
    img = cv2.imread(str(p))
    g = crop_black(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    v = cv2.resize(g, (64, 64), interpolation=cv2.INTER_AREA).astype(np.float32).flatten()
    return (v - v.mean()) / (v.std() + 1e-6)


def thong_tin_nhan(lbl):
    """Số polyp và tổng diện tích polygon (tỉ lệ so với ảnh) từ nhãn YOLO-seg."""
    so, dt = 0, 0.0
    for line in lbl.read_text().splitlines():
        parts = line.split()
        if len(parts) > 5:
            c = np.array(parts[1:], np.float64).reshape(-1, 2)
            x, y = c[:, 0], c[:, 1]
            dt += 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
            so += 1
    return so, dt


def doc_nguon():
    rows = []
    for split in ("train", "val"):
        for p in sorted((SRC / "images" / split).glob("*.jpg")):
            so, dt = thong_tin_nhan(SRC / "labels" / split / f"{p.stem}.txt")
            rows.append({"file": p.name, "split_cu": split, "is_bg": p.name.startswith("bg_"),
                         "so_polyp": so, "dien_tich_polyp": dt, "md5": hashlib.md5(p.read_bytes()).hexdigest()})
    df = pd.DataFrame(rows)
    assert (df.groupby(["split_cu", "is_bg"]).size().to_dict() ==
            {("train", False): 880, ("train", True): 160, ("val", False): 120, ("val", True): 40})
    assert (df[~df.is_bg].so_polyp > 0).all() and (df[df.is_bg].so_polyp == 0).all()
    return df


def cum_gan_trung(df, F):
    """Union-find các cặp tương quan >= CORR_DUP (chỉ trong cùng loại polyp/nền của train cũ)."""
    cha = list(range(len(df)))

    def goc(i):
        while cha[i] != i:
            cha[i] = cha[cha[i]]
            i = cha[i]
        return i

    idx = np.where(df.split_cu.values == "train")[0]
    C = F[idx] @ F[idx].T / F.shape[1]
    np.fill_diagonal(C, -1)
    cap = []
    for a, b in zip(*np.where(np.triu(C >= CORR_DUP))):
        i, j = idx[a], idx[b]
        cap.append((df.file[i], df.file[j], float(C[a, b])))
        cha[goc(i)] = goc(j)
    return np.array([goc(i) for i in range(len(df))]), cap


def chia(df, cum):
    rng = np.random.default_rng(SEED_CHIA)
    df = df.assign(cum=cum, split_moi="")
    df.loc[df.split_cu == "val", "split_moi"] = "test"
    tr = df[df.split_cu == "train"]
    # Tertile diện tích trên 880 ảnh polyp train cũ
    pol = tr[~tr.is_bg]
    bien = np.quantile(pol.dien_tich_polyp, [1 / 3, 2 / 3])
    df["nhom_kich_thuoc"] = np.where(df.is_bg, "nen",
                                     np.select([df.dien_tich_polyp <= bien[0], df.dien_tich_polyp <= bien[1]],
                                               ["nho", "vua"], "lon"))
    chon = set()
    # Polyp: mỗi nhóm kích thước lấy 40 ảnh, chọn theo cụm (xáo thứ tự cụm, cộng dồn đến đủ 40)
    for nhom in ("nho", "vua", "lon"):
        g = df[(df.split_cu == "train") & (df.nhom_kich_thuoc == nhom)]
        cums = g.cum.unique()
        rng.shuffle(cums)
        dem = 0
        for c in cums:
            thanh_vien = df.index[(df.cum == c) & (df.split_cu == "train")]
            if dem + len(thanh_vien) > N_VAL_POLYP // 3:
                continue
            chon.update(thanh_vien)
            dem += len(thanh_vien)
            if dem == N_VAL_POLYP // 3:
                break
        assert dem == N_VAL_POLYP // 3, nhom
    # Nền: 40 ảnh, cũng theo cụm
    g = df[(df.split_cu == "train") & df.is_bg]
    cums = g.cum.unique()
    rng.shuffle(cums)
    dem = 0
    for c in cums:
        thanh_vien = df.index[(df.cum == c) & (df.split_cu == "train")]
        if dem + len(thanh_vien) > N_VAL_NEN:
            continue
        chon.update(thanh_vien)
        dem += len(thanh_vien)
        if dem == N_VAL_NEN:
            break
    assert dem == N_VAL_NEN
    df.loc[sorted(chon), "split_moi"] = "val"
    df.loc[(df.split_cu == "train") & (df.split_moi == ""), "split_moi"] = "train"
    return df


def kiem_tra(df, F, cap_train):
    """md5 không lặp giữa các tập; ảnh gần trùng giữa mọi cặp tập (chỉ báo cáo, không đổi chia)."""
    loi = []
    dup = df[df.md5.duplicated(keep=False)]
    if len(dup):
        loi.append(f"md5 trùng: {dup.file.tolist()}")
    rows = []
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        ia, ib = np.where(df.split_moi == a)[0], np.where(df.split_moi == b)[0]
        C = F[ia] @ F[ib].T / F.shape[1]
        for x, y in zip(*np.where(C >= 0.90)):
            rows.append({"tap_1": a, "anh_1": df.file[ia[x]], "tap_2": b, "anh_2": df.file[ib[y]],
                         "tuong_quan": round(float(C[x, y]), 4)})
    kc = pd.DataFrame(rows, columns=["tap_1", "anh_1", "tap_2", "anh_2", "tuong_quan"]).sort_values(
        "tuong_quan", ascending=False)
    tv = kc[(kc.tap_1 == "train") & (kc.tap_2 == "val") & (kc.tuong_quan >= CORR_DUP)]
    if len(tv):
        loi.append(f"{len(tv)} cặp gần trùng train–val >= {CORR_DUP}")
    return kc, loi


def chep(df):
    if OUT.exists():
        raise SystemExit(f"{OUT} đã tồn tại. Không ghi đè; chuyển thư mục cũ ra _Luu_Tru_Ngoai nếu muốn tạo lại.")
    for r in df.itertuples():
        for loai, duoi in (("images", ".jpg"), ("labels", ".txt")):
            src = SRC / loai / r.split_cu / (Path(r.file).stem + duoi)
            dst = OUT / loai / r.split_moi / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
    (OUT / "data.yaml").write_text(
        "# Hướng 2: BG20 chia 3 tập (xem ../danh_sach_chia.csv). Notebook Kaggle tự ghi lại 'path'.\n"
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n\nnames:\n  0: polyp\n", encoding="utf-8")
    # Kiểm tra lại sau khi chép: đếm file, md5 khớp nguồn
    for s in ("train", "val", "test"):
        n_img = len(list((OUT / "images" / s).glob("*.jpg")))
        n_lbl = len(list((OUT / "labels" / s).glob("*.txt")))
        assert n_img == n_lbl == (df.split_moi == s).sum(), s
    for r in df.sample(50, random_state=0).itertuples():
        assert hashlib.md5((OUT / "images" / r.split_moi / r.file).read_bytes()).hexdigest() == r.md5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chi-kiem-tra", action="store_true", help="chia và kiểm tra, không chép ảnh")
    args = ap.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    df = doc_nguon()
    F = np.stack([dac_trung(SRC / "images" / r.split_cu / r.file) for r in df.itertuples()])
    cum, cap = cum_gan_trung(df, F)
    df = chia(df, cum)
    kc, loi = kiem_tra(df, F, cap)

    bang = df.groupby(["split_moi", "nhom_kich_thuoc"]).size().unstack(fill_value=0)
    print(bang.loc[["train", "val", "test"], ["nho", "vua", "lon", "nen"]].to_string())
    print(f"Cặp gần trùng trong train cũ (>= {CORR_DUP}): {len(cap)}; số cụm >1 ảnh: "
          f"{int((pd.Series(cum).value_counts() > 1).sum())}")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        k = kc[(kc.tap_1 == a) & (kc.tap_2 == b)]
        print(f"  {a}–{b}: >= 0,95: {int((k.tuong_quan >= 0.95).sum())}, >= 0,90: {len(k)}")
    print("Lỗi:", loi or "không")
    assert not loi

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    cot = ["file", "split_cu", "split_moi", "is_bg", "so_polyp", "dien_tich_polyp", "nhom_kich_thuoc", "cum", "md5"]
    if args.chi_kiem_tra:
        return
    df[cot].to_csv(OUT_ROOT / "danh_sach_chia.csv", index=False, encoding="utf-8-sig")
    kc.to_csv(OUT_ROOT / "kiem_tra_gan_trung.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(cap, columns=["anh_1", "anh_2", "tuong_quan"]).to_csv(
        OUT_ROOT / "cap_gan_trung_trong_train_cu.csv", index=False, encoding="utf-8-sig")
    chep(df)
    tom = {"seed_chia": SEED_CHIA, "nguong_gan_trung": CORR_DUP, "nguon": str(SRC.relative_to(TEST_MAU.parent)),
           "so_anh": {s: {"polyp": int(((df.split_moi == s) & ~df.is_bg).sum()),
                          "nen": int(((df.split_moi == s) & df.is_bg).sum()),
                          "so_polyp": int(df[df.split_moi == s].so_polyp.sum())} for s in ("train", "val", "test")},
           "phan_tang_val": bang.loc["val"].to_dict()}
    (OUT_ROOT / "tom_tat_chia.json").write_text(json.dumps(tom, ensure_ascii=False, indent=2, default=int), encoding="utf-8")
    print(json.dumps(tom, ensure_ascii=False, indent=1, default=int))


if __name__ == "__main__":
    main()
