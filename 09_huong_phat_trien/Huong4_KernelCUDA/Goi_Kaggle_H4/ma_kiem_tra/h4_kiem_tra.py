"""Hướng 4, mốc 2: kiểm tra kernel selective_scan thật và đo tốc độ (chạy trên Kaggle T4, chạy thử được trên laptop).

Gói ultralytics_TSVM_CUDA phải được import (tên gói "ultralytics") trước khi gọi các hàm ở đây.
Cách quét chọn bằng biến môi trường TSVM_SCAN ("pytorch" | "cuda"), đọc ở mỗi lần forward,
nên cùng một mô hình TSVM được đo bằng cả hai cách.

Các phần:
  A  kernel thật so với PyTorch: SS2D ngẫu nhiên (forward + gradient), tầng 10 với đặc trưng thật của ảnh val,
     đầu ra toàn mô hình.
  B  đánh giá 160 ảnh val bằng best.pt seed 0 (baseline, TSVM-PyTorch, TSVM-kernel).
  C  độ trễ forward (batch 1, FP32, chỉ mô hình): 5 vòng xen kẽ, đổi thứ tự mỗi vòng, CUDA event.
  D  độ trễ riêng khối SS2D (batch 1 và 8).
  E  một bước huấn luyện (batch 8, forward + backward + AdamW) và bộ nhớ GPU đỉnh.
  F  "video": dự đoán lần lượt từng ảnh val như các khung hình liên tiếp (gồm tiền xử lý và hậu xử lý), đo FPS.
"""
from __future__ import annotations

import gc
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

CAU_HINH = [("baseline", None), ("tsvm_pytorch", "pytorch"), ("tsvm_cuda", "cuda")]


def log(*a):
    print(*a, flush=True)


def dat_che_do(mode):
    if mode is not None:
        os.environ["TSVM_SCAN"] = mode


def gpu_state():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=10).stdout.splitlines()[0].split(",")
        return float(out[0]), float(out[1])
    except Exception:
        return np.nan, np.nan


def don_dep():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ------------------------------------------------------------------ selective scan viết tay (chỉ để chạy thử trên laptop)
def selective_scan_ref(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False, return_last_state=False):
    """Quét tuần tự theo quy ước selective_scan_fn của mamba_ssm (u, delta: (b, dim, L); A: (dim, N); B, C: (b, G, N, L))."""
    dtype_in = u.dtype
    ct = torch.float64 if u.dtype == torch.float64 else torch.float32
    u, delta, A, B, C = u.to(ct), delta.to(ct), A.to(ct), B.to(ct), C.to(ct)
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].to(ct)
    if delta_softplus:
        delta = F.softplus(delta)
    b, dim, L = u.shape
    if B.dim() == 3:
        B = B.unsqueeze(1)
    if C.dim() == 3:
        C = C.unsqueeze(1)
    Bx = B.repeat_interleave(dim // B.shape[1], dim=1)
    Cx = C.repeat_interleave(dim // C.shape[1], dim=1)
    dA = torch.exp(delta.unsqueeze(2) * A[None, :, :, None])
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


# ------------------------------------------------------------------ tiện ích
def doc_anh(paths, imgsz=640):
    import cv2

    xs = []
    for p in paths:
        im = cv2.imread(str(p))[:, :, ::-1]
        im = cv2.resize(im, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        xs.append(torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255.0)
    return torch.stack(xs)


def cac_tensor(o):
    if isinstance(o, torch.Tensor):
        return [o]
    if isinstance(o, dict):
        return [t for v in o.values() for t in cac_tensor(v)]
    if isinstance(o, (list, tuple)):
        return [t for v in o for t in cac_tensor(v)]
    return []


def sai_khac_tuong_doi(a, b):
    ta, tb = cac_tensor(a), cac_tensor(b)
    assert len(ta) == len(tb)
    return max(((x.float() - y.float()).abs().max() / (x.float().abs().max() + 1e-12)).item() for x, y in zip(ta, tb))


# ------------------------------------------------------------------ A
def phan_A(tsvm_pt, anh_val, ket_qua):
    from ultralytics import YOLO
    import ultralytics.nn.modules.topology_shape_vmamba as tsv

    dev = "cuda"
    rows = []
    for shp in [(1, 256, 20, 20), (8, 256, 20, 20), (2, 64, 13, 17)]:
        torch.manual_seed(0)
        m = tsv.SS2D(d_model=shp[1]).to(dev).train()
        x = torch.randn(*shp, device=dev)
        x1, x2 = x.clone().requires_grad_(), x.clone().requires_grad_()
        y1 = m._forward_pytorch(x1)
        y1.square().mean().backward()
        g1 = {n: p.grad.clone() for n, p in m.named_parameters()}
        m.zero_grad()
        y2 = m._forward_cuda(x2)
        y2.square().mean().backward()
        g2 = {n: p.grad.clone() for n, p in m.named_parameters()}
        r = {"phan": "A1_SS2D_ngau_nhien", "kich_thuoc": str(shp), "sai_khac_dau_ra": sai_khac_tuong_doi(y1, y2),
             "sai_khac_grad_x": sai_khac_tuong_doi(x1.grad, x2.grad),
             "sai_khac_grad_tham_so": max(sai_khac_tuong_doi(g1[n], g2[n]) for n in g1)}
        rows.append(r)
        log(f"A1 SS2D {shp}: đầu ra {r['sai_khac_dau_ra']:.1e}, grad x {r['sai_khac_grad_x']:.1e}, grad tham số {r['sai_khac_grad_tham_so']:.1e}")

    # Đặc trưng thật của 8 ảnh val: so đầu ra khối SS2D và đầu ra khối TSVMamba bên trong tầng 10.
    # (Không so từng phần tử đầu ra cuối của mô hình: đầu YOLO26 chọn top-k theo điểm, sai khác rất nhỏ cũng đổi thứ tự;
    #  toàn mô hình được so bằng mAP ở phần B.)
    net = YOLO(str(tsvm_pt)).model.float().eval().to(dev)
    x = doc_anh(anh_val[:8]).to(dev)
    tsvm_blk = net.model[10].m[0]
    vao = {}
    h = tsvm_blk.register_forward_hook(lambda mod, inp, out: vao.setdefault("x", inp[0].detach()))
    with torch.inference_mode():
        dat_che_do("pytorch")
        net(x)
        h.remove()
        f = vao["x"]
        s_pt, b_pt = tsvm_blk.vmamba(f), tsvm_blk.ffn(tsvm_blk.fuse(torch.cat([tsvm_blk.vmamba(f) * (1 + tsvm_blk.gate(tsvm_blk.directional_extractor(tsvm_blk.shape_branch(f)))), tsvm_blk.directional_extractor(tsvm_blk.shape_branch(f))], 1)))
        dat_che_do("cuda")
        s_cu, b_cu = tsvm_blk.vmamba(f), tsvm_blk.ffn(tsvm_blk.fuse(torch.cat([tsvm_blk.vmamba(f) * (1 + tsvm_blk.gate(tsvm_blk.directional_extractor(tsvm_blk.shape_branch(f)))), tsvm_blk.directional_extractor(tsvm_blk.shape_branch(f))], 1)))
    r_s = {"phan": "A2_SS2D_dac_trung_that", "kich_thuoc": str(tuple(f.shape)), "sai_khac_dau_ra": sai_khac_tuong_doi(s_pt, s_cu)}
    r_b = {"phan": "A3_nhanh_TSVMamba_truoc_gamma", "kich_thuoc": str(tuple(f.shape)), "sai_khac_dau_ra": sai_khac_tuong_doi(b_pt, b_cu)}
    rows += [r_s, r_b]
    log(f"A2 SS2D tầng 10, đặc trưng thật 8 ảnh val: sai khác tương đối {r_s['sai_khac_dau_ra']:.1e}")
    log(f"A3 nhánh TSVMamba (trước gamma) tầng 10: sai khác tương đối {r_b['sai_khac_dau_ra']:.1e}")
    ket_qua["A"] = rows
    del net
    don_dep()
    worst = max(max(r.get("sai_khac_dau_ra", 0), r.get("sai_khac_grad_x", 0), r.get("sai_khac_grad_tham_so", 0)) for r in rows)
    log(f"A  sai khác lớn nhất: {worst:.1e} -> {'PASS' if worst < 1e-3 else 'CẦN XEM LẠI'} (ngưỡng 1e-3)")
    return rows


# ------------------------------------------------------------------ B
def phan_B(base_pt, tsvm_pt, data_yaml, out_dir, ket_qua):
    from ultralytics import YOLO

    rows = []
    for key, mode in CAU_HINH:
        dat_che_do(mode)
        mdl = YOLO(str(base_pt if key == "baseline" else tsvm_pt))
        t0 = time.time()
        res = mdl.val(data=str(data_yaml), split="val", imgsz=640, batch=8, device=0, half=False, plots=False,
                      verbose=False, project=str(out_dir / "val_runs"), name=key, exist_ok=True, workers=2)
        r = {"cau_hinh": key, "mask_map50_95": res.seg.map, "mask_map50": res.seg.map50, "mask_P": res.seg.mp,
             "mask_R": res.seg.mr, "box_map50_95": res.box.map, "giay": time.time() - t0}
        rows.append(r)
        log(f"B  {key:13s}: mask mAP50-95 {r['mask_map50_95']:.4f}, mAP50 {r['mask_map50']:.4f}, box {r['box_map50_95']:.4f}")
        del mdl, res
        don_dep()
    d = abs(rows[1]["mask_map50_95"] - rows[2]["mask_map50_95"])
    log(f"B  |mAP50-95 PyTorch - kernel| = {d:.5f} -> {'PASS' if d < 1e-3 else 'CẦN XEM LẠI'}")
    pd.DataFrame(rows).to_csv(out_dir / "B_val_160_anh.csv", index=False)
    ket_qua["B"] = rows


# ------------------------------------------------------------------ C
def phan_C(base_pt, tsvm_pt, out_dir, ket_qua, warmup=50, iters=200, rounds=5):
    from ultralytics import YOLO

    torch.backends.cudnn.benchmark = False
    base = YOLO(str(base_pt)).model.fuse().float().eval().cuda()
    tsvm = YOLO(str(tsvm_pt)).model.fuse().float().eval().cuda()
    nets = {"baseline": (base, None), "tsvm_pytorch": (tsvm, "pytorch"), "tsvm_cuda": (tsvm, "cuda")}
    x = torch.zeros(1, 3, 640, 640, device="cuda")
    rows, blocks = [], []
    with torch.inference_mode():
        for r in range(rounds):
            for key in (list(nets) if r % 2 == 0 else list(nets)[::-1]):
                net, mode = nets[key]
                dat_che_do(mode)
                t0c, c0 = gpu_state()
                for _ in range(warmup):
                    net(x)
                torch.cuda.synchronize()
                for i in range(iters):
                    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    s.record()
                    net(x)
                    e.record()
                    torch.cuda.synchronize()
                    rows.append({"cau_hinh": key, "vong": r, "lan": i, "ms": s.elapsed_time(e)})
                t1c, c1 = gpu_state()
                med = float(np.median([q["ms"] for q in rows[-iters:]]))
                blocks.append({"cau_hinh": key, "vong": r, "nhiet_truoc": t0c, "nhiet_sau": t1c, "xung_truoc": c0, "xung_sau": c1, "trung_vi_ms": med})
                log(f"C  vòng {r} {key:13s}: trung vị {med:.2f} ms ({t0c:.0f}->{t1c:.0f}°C, {c0:.0f}->{c1:.0f} MHz)")
    lat = pd.DataFrame(rows)
    lat.to_csv(out_dir / "C_do_tre_forward_tung_lan.csv", index=False)
    pd.DataFrame(blocks).to_csv(out_dir / "C_do_tre_forward_tung_khoi.csv", index=False)
    g = lat.groupby("cau_hinh")["ms"]
    tong = pd.DataFrame({"trung_vi_ms": g.median(), "p10_ms": g.quantile(0.10), "min_ms": g.min(), "tb_ms": g.mean()})
    blk = pd.DataFrame(blocks).pivot(index="vong", columns="cau_hinh", values="trung_vi_ms")
    tong.loc["tsvm_pytorch", "ti_le_so_baseline"] = (blk["tsvm_pytorch"] / blk["baseline"]).median()
    tong.loc["tsvm_cuda", "ti_le_so_baseline"] = (blk["tsvm_cuda"] / blk["baseline"]).median()
    tong.loc["tsvm_cuda", "ti_le_cuda_so_pytorch"] = (blk["tsvm_cuda"] / blk["tsvm_pytorch"]).median()
    tong.to_csv(out_dir / "C_do_tre_forward_tong_hop.csv")
    log(tong.round(3).to_string())
    ket_qua["C"] = tong.reset_index().to_dict(orient="records")
    del base, tsvm, nets
    don_dep()


# ------------------------------------------------------------------ D
def phan_D(tsvm_pt, out_dir, ket_qua, warmup=50, iters=200, rounds=3):
    from ultralytics import YOLO

    ss2d = YOLO(str(tsvm_pt)).model.model[10].m[0].vmamba.float().eval().cuda()
    rows = []
    with torch.inference_mode():
        for bs in (1, 8):
            x = torch.randn(bs, 256, 20, 20, device="cuda")
            for r in range(rounds):
                for mode in (("pytorch", "cuda") if r % 2 == 0 else ("cuda", "pytorch")):
                    dat_che_do(mode)
                    for _ in range(warmup):
                        ss2d(x)
                    torch.cuda.synchronize()
                    ts = []
                    for _ in range(iters):
                        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        s.record()
                        ss2d(x)
                        e.record()
                        torch.cuda.synchronize()
                        ts.append(s.elapsed_time(e))
                    rows.append({"batch": bs, "vong": r, "cach_quet": mode, "trung_vi_ms": float(np.median(ts))})
    d = pd.DataFrame(rows)
    tong = d.groupby(["batch", "cach_quet"])["trung_vi_ms"].median().unstack()
    tong["nhanh_hon_lan"] = tong["pytorch"] / tong["cuda"]
    d.to_csv(out_dir / "D_do_tre_ss2d.csv", index=False)
    log("D  khối SS2D (trung vị ms):\n" + tong.round(3).to_string())
    ket_qua["D"] = tong.reset_index().to_dict(orient="records")
    del ss2d
    don_dep()


# ------------------------------------------------------------------ E
def phan_E(base_pt, tsvm_pt, out_dir, ket_qua, batch=8, warmup=10, iters=40, rounds=3):
    from ultralytics import YOLO

    rows = []
    x = torch.rand(batch, 3, 640, 640, device="cuda")
    for r in range(rounds):
        for key, mode in (CAU_HINH if r % 2 == 0 else CAU_HINH[::-1]):
            dat_che_do(mode)
            net = YOLO(str(base_pt if key == "baseline" else tsvm_pt)).model.float().cuda().train()
            for p in net.parameters():
                p.requires_grad_(True)
            opt = torch.optim.AdamW(net.parameters(), lr=1e-5)
            don_dep()
            torch.cuda.reset_peak_memory_stats()

            def buoc():
                opt.zero_grad(set_to_none=True)
                loss = sum(t.float().mean() for t in cac_tensor(net(x)) if t.is_floating_point())
                loss.backward()
                opt.step()

            for _ in range(warmup):
                buoc()
            torch.cuda.synchronize()
            ts = []
            for _ in range(iters):
                s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                s.record()
                buoc()
                e.record()
                torch.cuda.synchronize()
                ts.append(s.elapsed_time(e))
            mem = torch.cuda.max_memory_allocated() / 2**30
            rows.append({"cau_hinh": key, "vong": r, "batch": batch, "ms_moi_buoc": float(np.median(ts)), "bo_nho_dinh_GB": mem})
            log(f"E  vòng {r} {key:13s}: {np.median(ts):.1f} ms/bước (batch {batch}), bộ nhớ đỉnh {mem:.2f} GB")
            del net, opt
            don_dep()
    d = pd.DataFrame(rows)
    d.to_csv(out_dir / "E_buoc_huan_luyen.csv", index=False)
    tong = d.groupby("cau_hinh")[["ms_moi_buoc", "bo_nho_dinh_GB"]].median()
    log("E  tổng hợp (trung vị):\n" + tong.round(3).to_string())
    ket_qua["E"] = tong.reset_index().to_dict(orient="records")


# ------------------------------------------------------------------ F
def phan_F(base_pt, tsvm_pt, anh_val, out_dir, ket_qua, rounds=3):
    from ultralytics import YOLO

    mdl = {"baseline": YOLO(str(base_pt)), "tsvm": YOLO(str(tsvm_pt))}
    for m in mdl.values():  # khởi tạo predictor + gộp Conv-BN trước khi đo
        m.predict(str(anh_val[0]), imgsz=640, conf=0.25, iou=0.7, retina_masks=True, device=0, half=False, verbose=False)
    rows = []
    for r in range(rounds):
        for key, mode in (CAU_HINH if r % 2 == 0 else CAU_HINH[::-1]):
            dat_che_do(mode)
            m = mdl["baseline" if key == "baseline" else "tsvm"]
            for p in anh_val[:10]:
                m.predict(str(p), imgsz=640, conf=0.25, iou=0.7, retina_masks=True, device=0, half=False, verbose=False)
            torch.cuda.synchronize()
            per, sp = [], []
            t0 = time.perf_counter()
            for p in anh_val:
                t1 = time.perf_counter()
                res = m.predict(str(p), imgsz=640, conf=0.25, iou=0.7, retina_masks=True, device=0, half=False, verbose=False)
                if res[0].masks is not None:
                    res[0].masks.data.cpu()  # đưa mặt nạ về CPU như khi vẽ lên khung hình
                torch.cuda.synchronize()
                per.append((time.perf_counter() - t1) * 1000)
                sp.append(res[0].speed)
            tong = time.perf_counter() - t0
            rows.append({"cau_hinh": key, "vong": r, "so_khung": len(anh_val), "fps": len(anh_val) / tong,
                         "ms_moi_khung_trung_vi": float(np.median(per)),
                         "tien_xu_ly_ms": float(np.median([s["preprocess"] for s in sp])),
                         "suy_luan_ms": float(np.median([s["inference"] for s in sp])),
                         "hau_xu_ly_ms": float(np.median([s["postprocess"] for s in sp]))})
            log(f"F  vòng {r} {key:13s}: {rows[-1]['fps']:.1f} khung/giây, {rows[-1]['ms_moi_khung_trung_vi']:.1f} ms/khung "
                f"(suy luận {rows[-1]['suy_luan_ms']:.1f} ms)")
    d = pd.DataFrame(rows)
    d.to_csv(out_dir / "F_video_160_khung.csv", index=False)
    tong = d.groupby("cau_hinh")[["fps", "ms_moi_khung_trung_vi", "tien_xu_ly_ms", "suy_luan_ms", "hau_xu_ly_ms"]].median()
    log("F  tổng hợp (trung vị 3 vòng):\n" + tong.round(2).to_string())
    ket_qua["F"] = tong.reset_index().to_dict(orient="records")
    del mdl
    don_dep()


def thong_tin_moi_truong():
    import ultralytics
    import ultralytics.nn.modules.topology_shape_vmamba as tsv

    info = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "torch": torch.__version__,
            "cuda": torch.version.cuda, "ultralytics": ultralytics.__version__, "ultralytics_tu": ultralytics.__file__,
            "kernel_mamba_ssm": tsv.cuda_kernel_available(), "kernel_la_ban_viet_tay": tsv._selective_scan_fn is selective_scan_ref,
            "python": platform.python_version(), "may": platform.platform()}
    try:
        import mamba_ssm

        info["mamba_ssm"] = mamba_ssm.__version__
    except Exception as e:
        info["mamba_ssm"] = f"không có ({type(e).__name__})"
    return info


def chay_tat_ca(base_pt, tsvm_pt, val_dir, data_yaml, out_dir, nhanh=False):
    """nhanh=True: số lần đo nhỏ, dùng khi chạy thử trên laptop."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    anh_val = sorted(Path(val_dir).glob("*.jpg")) + sorted(Path(val_dir).glob("*.png"))
    assert len(anh_val) == 160, len(anh_val)
    ket_qua = {"moi_truong": thong_tin_moi_truong(), "nhanh": nhanh}
    log(json.dumps(ket_qua["moi_truong"], ensure_ascii=False, indent=1))
    t0 = time.time()
    try:
        phan_A(tsvm_pt, anh_val, ket_qua)
        phan_B(base_pt, tsvm_pt, data_yaml, out_dir, ket_qua)
        if nhanh:
            phan_C(base_pt, tsvm_pt, out_dir, ket_qua, warmup=3, iters=5, rounds=2)
            phan_D(tsvm_pt, out_dir, ket_qua, warmup=2, iters=3, rounds=2)
            phan_E(base_pt, tsvm_pt, out_dir, ket_qua, batch=2, warmup=1, iters=2, rounds=1)
            phan_F(base_pt, tsvm_pt, anh_val[:20], out_dir, ket_qua, rounds=1)
        else:
            phan_C(base_pt, tsvm_pt, out_dir, ket_qua)
            phan_D(tsvm_pt, out_dir, ket_qua)
            phan_E(base_pt, tsvm_pt, out_dir, ket_qua)
            phan_F(base_pt, tsvm_pt, anh_val, out_dir, ket_qua)
    finally:
        ket_qua["phut"] = (time.time() - t0) / 60
        (out_dir / "ket_qua_moc2.json").write_text(json.dumps(ket_qua, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        log(f"Ghi {out_dir / 'ket_qua_moc2.json'} ({ket_qua['phut']:.1f} phút)")
    return ket_qua
