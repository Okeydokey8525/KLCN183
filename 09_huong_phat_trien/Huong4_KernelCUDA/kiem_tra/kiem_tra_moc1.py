"""Hướng 4, mốc 1: kiểm tra bản SS2D hai nhánh (PyTorch / kernel CUDA) trên laptop, chưa cần mamba_ssm.

Các phép kiểm tra:
  T1  Chỉ tệp nn/modules/topology_shape_vmamba.py khác bản TSVM_FP32 của báo cáo (Test_Mau chỉ đọc).
  T2  Nhánh PyTorch của bản mới cho kết quả TRÙNG BIT với bản gốc (forward + gradient), trên CPU và GPU.
  T3  Hàm quét song song Hillis-Steele khớp với quét tuần tự (float64), tức thuật toán gốc đúng.
  T4  Nhánh kernel (_forward_cuda) chạy với một bản selective_scan viết tay bằng PyTorch, cùng quy ước
      tham số với mamba_ssm.selective_scan_fn, cho kết quả khớp nhánh PyTorch => cách xếp 4 hướng,
      softplus, A, B, C đúng. Kernel thật sẽ được thử ở mốc 2 trên Kaggle.
  T5  TSVM_SCAN=cuda báo lỗi rõ khi không có mamba_ssm; auto tự quay về PyTorch.
  T6  Nạp best.pt seed 0 của báo cáo, đánh giá 160 ảnh val bằng nhánh PyTorch và nhánh kernel giả lập:
      mAP phải trùng nhau; so thêm với val mAP50-95(M) = 0,7245 ghi trong results.csv lúc huấn luyện.

Chạy: C:\\venv_kl_gpu\\Scripts\\python.exe -X utf8 kiem_tra_moc1.py   (dùng GPU laptop vài phút)
Ghi log: kiem_tra/ket_qua_moc1.txt
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
H4 = HERE.parent
DL = H4.parents[1]
FORK_GOC = DL / "Test_Mau" / "02_ma_nguon" / "ultralytics_TSVM_FP32"
FORK_H4 = H4 / "ma_nguon" / "ultralytics_TSVM_CUDA"
BEST_S0 = DL / "Test_Mau" / "03_ket_qua_huan_luyen" / "TSVM_FP32" / "Kvasir_BG20_YOLO26s_seg_TSVM_FP32_s0_w2" / "weights" / "best.pt"
DATA_BG20 = DL / "Test_Mau" / "01_du_lieu" / "Kvasir_YOLO_SEG_BG20"
LOG = HERE / "ket_qua_moc1.txt"
REL = "nn/modules/topology_shape_vmamba.py"

_lines: list[str] = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _lines.append(s)


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


# ---------------------------------------------------------------- T1
def t1_chi_mot_tep_khac():
    goc = {p.relative_to(FORK_GOC).as_posix(): p for p in FORK_GOC.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    moi = {p.relative_to(FORK_H4).as_posix(): p for p in FORK_H4.rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    assert set(goc) == set(moi), f"Danh sách tệp khác nhau: {set(goc) ^ set(moi)}"
    khac = sorted(k for k in goc if md5(goc[k]) != md5(moi[k]))
    log(f"T1  {len(goc)} tệp, khác nhau: {khac}")
    assert khac == [REL], khac
    log("T1  PASS: chỉ topology_shape_vmamba.py thay đổi")


# ---------------------------------------------------------------- nạp gói
def import_fork():
    tmp = Path(tempfile.mkdtemp(prefix="tsvm_h4_"))
    link = tmp / "ultralytics"
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(FORK_H4)], check=True, capture_output=True)
    sys.path.insert(0, str(tmp))
    import ultralytics  # noqa: F401

    assert Path(ultralytics.__file__).resolve().parent == FORK_H4.resolve(), ultralytics.__file__
    return tmp


def nap_module_goc():
    """Nạp tệp topology_shape_vmamba.py GỐC (Test_Mau) như một module con của gói đang dùng (để `.conv` giải được)."""
    spec = importlib.util.spec_from_file_location("ultralytics.nn.modules._tsvm_goc_baocao", FORK_GOC / REL)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------- selective scan viết tay (quy ước mamba_ssm)
def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """Quét tuần tự theo đúng quy ước selective_scan_fn của mamba_ssm.

    u, delta: (b, dim, L); A: (dim, N); B, C: (b, G, N, L) hoặc (b, N, L); D: (dim,); z: (b, dim, L).
    """
    import torch
    import torch.nn.functional as F

    dtype_in = u.dtype
    ct = torch.float64 if u.dtype == torch.float64 else torch.float32  # kernel thật tính FP32; test float64 giữ float64
    u, delta, A = u.to(ct), delta.to(ct), A.to(ct)
    B, C = B.to(ct), C.to(ct)
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].to(ct)
    if delta_softplus:
        delta = F.softplus(delta)
    b, dim, L = u.shape
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    Bx = B.repeat_interleave(dim // B.shape[1], dim=1)  # (b, dim, N, L)
    Cx = C.repeat_interleave(dim // C.shape[1], dim=1)
    dA = torch.exp(delta.unsqueeze(2) * A[None, :, :, None])  # (b, dim, N, L)
    dBu = delta.unsqueeze(2) * Bx * u.unsqueeze(2)
    h = u.new_zeros(b, dim, A.shape[1])
    ys = []
    for t in range(L):
        h = dA[..., t] * h + dBu[..., t]
        ys.append((h * Cx[..., t]).sum(-1))
    y = torch.stack(ys, dim=-1)
    if D is not None:
        y = y + u * D[:, None].to(ct)
    if z is not None:
        y = y * F.silu(z.to(ct))
    return y.to(dtype_in)


# ---------------------------------------------------------------- T2
def t2_trung_bit(goc, moi, devices):
    import torch

    os.environ["TSVM_SCAN"] = "pytorch"
    for dev in devices:
        for cls_name, kw, shp in [("SS2D", dict(d_model=256), (2, 256, 20, 20)),
                                  ("C2TSVMamba", dict(c1=512, c2=512), (2, 512, 20, 20)),
                                  ("SS2D", dict(d_model=64), (3, 64, 13, 17))]:
            torch.manual_seed(0)
            a = getattr(goc, cls_name)(**kw).to(dev).train()
            b = getattr(moi, cls_name)(**kw).to(dev).train()
            b.load_state_dict(a.state_dict())
            x = torch.randn(*shp, device=dev)
            xa, xb = x.clone().requires_grad_(), x.clone().requires_grad_()
            ya, yb = a(xa), b(xb)
            ya.square().mean().backward()
            yb.square().mean().backward()
            same_y = torch.equal(ya, yb)
            same_gx = torch.equal(xa.grad, xb.grad)
            same_gp = all(torch.equal(pa.grad, pb.grad) for pa, pb in zip(a.parameters(), b.parameters()) if pa.grad is not None)
            log(f"T2  {dev:4s} {cls_name:10s} {tuple(shp)}: đầu ra trùng bit={same_y}, grad x={same_gx}, grad tham số={same_gp}")
            assert same_y and same_gx and same_gp
    log("T2  PASS: nhánh PyTorch y hệt bản báo cáo")


# ---------------------------------------------------------------- T3
def t3_quet_song_song(moi):
    import torch

    torch.manual_seed(1)
    for L in (1, 2, 7, 400, 513):
        a = torch.rand(2, 8, 4, L, dtype=torch.float64) * 0.98 + 0.01
        b = torch.randn(2, 8, 4, L, dtype=torch.float64)
        h_ss = moi.parallel_associative_scan(a, b)
        h = torch.zeros_like(b[..., 0])
        seq = []
        for t in range(L):
            h = a[..., t] * h + b[..., t]
            seq.append(h)
        h_seq = torch.stack(seq, -1)
        err = (h_ss - h_seq).abs().max().item()
        log(f"T3  L={L:4d}: sai khác lớn nhất so với quét tuần tự = {err:.2e}")
        assert err < 1e-10
    log("T3  PASS: Hillis-Steele = quét tuần tự")


# ---------------------------------------------------------------- T4
def t4_nhanh_kernel_gia_lap(moi, devices):
    import torch

    cu = moi._selective_scan_fn
    moi._selective_scan_fn = selective_scan_ref
    try:
        for dev in devices:
            for kw, shp in [(dict(d_model=256), (2, 256, 20, 20)), (dict(d_model=64), (3, 64, 13, 17))]:
                torch.manual_seed(0)
                m = moi.SS2D(**kw).to(dev).double()
                x = torch.randn(*shp, device=dev, dtype=torch.float64)
                x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
                y1 = m._forward_pytorch(x1)
                y1.square().mean().backward()
                g1 = [p.grad.clone() for p in m.parameters()]
                m.zero_grad()
                y2 = m._forward_cuda(x2)
                y2.square().mean().backward()
                g2 = [p.grad.clone() for p in m.parameters()]
                ey = ((y1 - y2).abs().max() / y1.abs().max()).item()
                eg = max(((a - b).abs().max() / (a.abs().max() + 1e-30)).item() for a, b in zip(g1, g2))
                egx = ((x1.grad - x2.grad).abs().max() / x1.grad.abs().max()).item()
                log(f"T4  {dev:4s} SS2D{tuple(shp)} float64: sai khác tương đối đầu ra={ey:.1e}, grad x={egx:.1e}, grad tham số={eg:.1e}")
                assert ey < 1e-9 and egx < 1e-9 and eg < 1e-9
    finally:
        moi._selective_scan_fn = cu
    log("T4  PASS: cách xếp 4 hướng cho kernel khớp phép tính gốc")


# ---------------------------------------------------------------- T5
def t5_chon_nhanh(moi):
    import torch

    m = moi.SS2D(d_model=32)
    x = torch.randn(1, 32, 5, 5)
    os.environ["TSVM_SCAN"] = "cuda"
    try:
        m(x)
        raise AssertionError("lẽ ra phải báo lỗi")
    except RuntimeError as e:
        log(f"T5  TSVM_SCAN=cuda không có mamba_ssm -> RuntimeError: {str(e)[:90]}")
    os.environ["TSVM_SCAN"] = "auto"
    assert torch.equal(m(x), m._forward_pytorch(x))
    os.environ["TSVM_SCAN"] = "sai"
    try:
        m(x)
        raise AssertionError("lẽ ra phải báo lỗi")
    except ValueError:
        pass
    os.environ["TSVM_SCAN"] = "pytorch"
    log(f"T5  mamba_ssm có sẵn trên laptop: {moi.cuda_kernel_available()}; auto -> PyTorch: đúng")
    log("T5  PASS")


# ---------------------------------------------------------------- T6
def t6_val_best_s0(moi):
    import pandas as pd
    import torch
    import yaml
    from ultralytics import YOLO

    tmp = Path(tempfile.mkdtemp(prefix="h4_val_"))
    for sub in ("images/val", "labels/val"):  # chép val sang thư mục tạm: Test_Mau chỉ đọc, val() sẽ ghi .cache
        shutil.copytree(DATA_BG20 / sub, tmp / sub)
    data_yaml = tmp / "data.yaml"
    data_yaml.write_text(yaml.safe_dump({"path": str(tmp), "train": "images/val", "val": "images/val", "nc": 1, "names": {0: "polyp"}}), encoding="utf-8")
    n_val = len(list((tmp / "images/val").iterdir()))

    r = pd.read_csv(BEST_S0.parents[1] / "results.csv")
    r.columns = [c.strip() for c in r.columns]
    fit = 0.1 * (r["metrics/mAP50(B)"] + r["metrics/mAP50(M)"]) + 0.9 * (r["metrics/mAP50-95(B)"] + r["metrics/mAP50-95(M)"])
    ref = r.loc[fit.idxmax()]
    log(f"T6  results.csv seed 0, epoch {int(ref['epoch'])}: mask mAP50-95={ref['metrics/mAP50-95(M)']:.4f}, mAP50={ref['metrics/mAP50(M)']:.4f}")

    out = {}
    cu = moi._selective_scan_fn
    for ten, mode, fn in [("PyTorch (Hillis-Steele)", "pytorch", cu), ("Kernel giả lập (quét tuần tự)", "cuda", selective_scan_ref)]:
        os.environ["TSVM_SCAN"] = mode
        moi._selective_scan_fn = fn
        model = YOLO(str(BEST_S0))
        l10 = model.model.model[10]
        n10 = sum(p.numel() for p in l10.parameters())
        ntot = sum(p.numel() for p in model.model.parameters())
        assert type(l10).__module__ == moi.__name__, type(l10).__module__
        t0 = time.time()
        res = model.val(data=str(data_yaml), split="val", imgsz=640, batch=8, device=0, half=False, plots=False,
                        verbose=False, project=str(tmp / "runs"), name=mode, workers=0)
        dt = time.time() - t0
        out[mode] = (res.seg.map, res.seg.map50, res.box.map)
        log(f"T6  {ten:30s}: {n_val} ảnh, tầng 10 = {n10:,} tham số, tổng {ntot:,}; mask mAP50-95={res.seg.map:.4f}, "
            f"mAP50={res.seg.map50:.4f}, box mAP50-95={res.box.map:.4f} ({dt:.0f} s)")
        del model
        torch.cuda.empty_cache()
    moi._selective_scan_fn = cu
    os.environ["TSVM_SCAN"] = "pytorch"
    d = max(abs(a - b) for a, b in zip(out["pytorch"], out["cuda"]))
    log(f"T6  chênh lớn nhất giữa hai nhánh: {d:.2e}")
    assert d < 5e-4
    log("T6  PASS: cùng best.pt, hai cách quét cho cùng mAP")
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    log(f"=== Hướng 4, mốc 1, {time.strftime('%Y-%m-%d %H:%M')} ===")
    t1_chi_mot_tep_khac()
    tmp = import_fork()
    import torch
    import ultralytics
    import ultralytics.nn.modules.topology_shape_vmamba as moi

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    log(f"torch {torch.__version__}, ultralytics {ultralytics.__version__} từ {Path(ultralytics.__file__).parent}, "
        f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'không'}")
    goc = nap_module_goc()
    try:
        t2_trung_bit(goc, moi, devices)
        t3_quet_song_song(moi)
        t4_nhanh_kernel_gia_lap(moi, devices)
        t5_chon_nhanh(moi)
        if torch.cuda.is_available():
            t6_val_best_s0(moi)
        log("=== TẤT CẢ PASS ===")
    finally:
        LOG.write_text("\n".join(_lines) + "\n", encoding="utf-8")
        subprocess.run(["cmd", "/c", "rmdir", str(tmp / "ultralytics")], capture_output=True)


if __name__ == "__main__":
    main()
