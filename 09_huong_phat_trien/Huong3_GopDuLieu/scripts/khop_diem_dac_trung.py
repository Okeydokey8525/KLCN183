"""Hướng 3: đo "cùng cảnh" giữa hai ảnh bằng khớp điểm đặc trưng ORB + RANSAC (số điểm khớp hợp lệ, inlier).

Tương quan ảnh xám 64×64 không đủ: hai khung liền nhau của cùng video (vd. PolypGen C2 001160 và 001161) chỉ có
tương quan 0,86, trong khi nhiều cặp ảnh Kvasir khác nhau cũng đạt 0,85–0,90. Khớp điểm đặc trưng bắt được hai khung
cùng cảnh dù camera dịch chuyển, và ít khớp nhầm giữa hai ảnh khác nhau.

Chỉ lấy điểm trong elip trung tâm (bỏ viền đen, chữ, hộp xanh của máy nội soi ở góc) để tránh khớp nhầm vào lớp phủ.
"""
from __future__ import annotations

import cv2
import numpy as np

ORB = cv2.ORB_create(nfeatures=1000, fastThreshold=10)
BF = cv2.BFMatcher(cv2.NORM_HAMMING)


def dac_trung(path: str, canh: int = 512):
    a = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE)
    s = canh / max(a.shape)
    a = cv2.resize(a, (round(a.shape[1] * s), round(a.shape[0] * s)), interpolation=cv2.INTER_AREA)
    ys, xs = np.where(a > 25)
    m = np.zeros_like(a)
    if len(ys):
        cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
        ry, rx = (ys.max() - ys.min()) * 0.40, (xs.max() - xs.min()) * 0.40
        cv2.ellipse(m, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 255, -1)
    a = cv2.createCLAHE(2.0, (8, 8)).apply(a)
    kp, des = ORB.detectAndCompute(a, m)
    pts = np.float32([k.pt for k in kp]) if kp else np.zeros((0, 2), np.float32)
    return pts, des


def inlier(f1, f2, ti_le=0.75) -> int:
    (p1, d1), (p2, d2) = f1, f2
    if d1 is None or d2 is None or len(d1) < 8 or len(d2) < 8:
        return 0
    good = []
    for mm in BF.knnMatch(d1, d2, k=2):
        if len(mm) == 2 and mm[0].distance < ti_le * mm[1].distance:
            good.append(mm[0])
    if len(good) < 8:
        return len(good) // 2
    a = p1[[g.queryIdx for g in good]]
    b = p2[[g.trainIdx for g in good]]
    H, msk = cv2.findHomography(a, b, cv2.RANSAC, 5.0)
    if H is None:
        return 0
    n = int(msk.sum())
    # bỏ phép biến đổi gần đồng nhất (lớp phủ đứng yên trùng nhau giữa hai ảnh khác nhau)
    if np.allclose(H, np.eye(3), atol=0.02) and n < 40:
        return 0
    return n


NGUONG_INLIER = 25   # hiệu chỉnh 19/11 trên 525 cặp đã soi bằng mắt: 499 cặp ảnh khác có tối đa 20 (99% <= 10),
                     # 24 cặp cùng ảnh 271–531, cặp cùng tổn thương khác khung 201; cặp C2 001160/001161 là 68
NGUONG_INLIER_CHAC = 50   # soi lần 2: khung mờ/đỏ của chuỗi âm tính khớp nhầm tới 43 inlier với ảnh khác hẳn,
NGUONG_CORR_KEM = 0.88    # nên 25–49 inlier chỉ tính là cùng cảnh khi tương quan 64×64 cũng >= 0,88


def la_cung_canh(corr, inl):
    import numpy as _np
    corr, inl = _np.asarray(corr), _np.asarray(inl)
    return (inl >= NGUONG_INLIER_CHAC) | ((inl >= NGUONG_INLIER) & (corr >= NGUONG_CORR_KEM))


def vec64(path: str) -> np.ndarray:
    a = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_GRAYSCALE).astype(np.float32)
    ys, xs = np.where(a > 20)
    if len(ys):
        a = a[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    v = cv2.resize(a, (64, 64), interpolation=cv2.INTER_AREA).ravel()
    return (v - v.mean()) / (v.std() + 1e-6)


def cap_cung_canh(paths: list[str], nhom_ten: list[str], so: list[float], nguong_corr: float = 0.70, nhom_bo=None):
    """Trả về DataFrame các cặp (i, j, corr, inlier) có inlier >= NGUONG_INLIER.

    Ứng viên: tương quan 64×64 >= nguong_corr, hoặc cùng nhom_ten (vd. cùng trung tâm PolypGen) và số thứ tự tệp
    cách nhau <= 5 (khung liền nhau của cùng video có thể tương quan thấp khi camera di chuyển).
    """
    from concurrent.futures import ThreadPoolExecutor
    import pandas as pd
    n = len(paths)
    with ThreadPoolExecutor(8) as ex:
        V = np.stack(list(ex.map(vec64, paths)))
        F = list(ex.map(dac_trung, paths))
    C = V @ V.T / V.shape[1]
    ung = set(map(tuple, np.argwhere(np.triu(C >= nguong_corr, 1))))
    so = np.asarray(so, float)
    for g in set(nhom_ten):
        if not g:
            continue
        idx = [i for i in range(n) if nhom_ten[i] == g and not np.isnan(so[i])]
        idx.sort(key=lambda i: so[i])
        for a in range(len(idx)):
            for b in range(a + 1, len(idx)):
                if so[idx[b]] - so[idx[a]] > 5:
                    break
                ung.add(tuple(sorted((idx[a], idx[b]))))
    if nhom_bo is not None:          # chỉ xét cặp thuộc hai bộ khác nhau
        ung = {(i, j) for i, j in ung if nhom_bo[i] != nhom_bo[j]}
    ung = sorted(ung)
    with ThreadPoolExecutor(8) as ex:
        nin = list(ex.map(lambda p: inlier(F[p[0]], F[p[1]]), ung))
    df = pd.DataFrame([(i, j, float(C[i, j]), k) for (i, j), k in zip(ung, nin)], columns=["i", "j", "corr", "inlier"])
    print(f"Khớp điểm: {len(ung)} cặp ứng viên, {int(la_cung_canh(df['corr'], df.inlier).sum())} cặp cùng cảnh")
    return df
