"""Dựng VM-UNet trên Windows (không có gói mamba_ssm) để SUY LUẬN checkpoint đã huấn luyện trên Kaggle.

mamba_ssm chỉ có kernel CUDA cho Linux. Mã gốc vmamba.py gọi hàm toàn cục `selective_scan_fn`, và lệnh import
mamba_ssm đã nằm trong try/except. Vì vậy chỉ cần gán một bản quét viết bằng PyTorch thuần vào tên đó.

Phép quét (đúng định nghĩa selective_scan_ref của mamba_ssm, z = None):
    delta = softplus(delta + delta_bias)
    x_t   = exp(delta_t * A) * x_{t-1} + delta_t * B_t * u_t
    y_t   = <C_t, x_t> + D * u_t
Ba cách cài, cùng chia chuỗi thành khối và mang trạng thái x giữa các khối:
- quet_theo_khoi: x_t = sum_{r<=t} exp(S_t - S_r) b_r (ma trận T×T), chính xác nhưng chậm.
- quet_log (float64) và quet_log32 (float32, dùng khi suy luận): x_t = exp(S_t) * cumsum(exp(-S_r) b_r)
  tính bằng logcumsumexp, tách b thành phần dương và âm. S là tổng tích lũy của delta*A (<= 0).

Kiểm chứng: `python vmunet_quet_thuan.py` so bản theo khối với bản tuần tự trên dữ liệu ngẫu nhiên.
Kiểm chứng đầu-cuối nằm trong danh_gia_pixel_vmunet.py: tính lại Val Dice của notebook và so với tom_tat.txt.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

GOC = Path(__file__).resolve().parent.parent                       # Baseline_PVT_VMUNet/Huong1_PVT_VMUNet
VMUNET_REPO = GOC / "ma_nguon_goc" / "VM-UNet-main"


def quet_tuan_tu(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                 return_last_state=False):
    """Bản tham chiếu, viết lại từ selective_scan_ref (dùng để kiểm tra, rất chậm)."""
    assert z is None and not return_last_state
    b, dd, L = u.shape
    G, N = B.shape[1], A.shape[1]
    if delta_bias is not None:
        delta = delta + delta_bias[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    Bx = B.repeat_interleave(dd // G, dim=1)                      # (b, dd, N, L)
    Cx = C.repeat_interleave(dd // G, dim=1)
    x = u.new_zeros(b, dd, N)
    ys = []
    for t in range(L):
        x = torch.exp(delta[:, :, t, None] * A) * x + (delta[:, :, t] * u[:, :, t])[..., None] * Bx[..., t]
        ys.append((x * Cx[..., t]).sum(-1))
    y = torch.stack(ys, -1)
    return y + u * D[None, :, None] if D is not None else y


def quet_theo_khoi(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
                   return_last_state=False, T=64):
    """Thay cho mamba_ssm.selective_scan_fn. u, delta: (b, G*d, L); A: (G*d, N); B, C: (b, G, N, L)."""
    assert z is None and not return_last_state
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    b, dd, L = u.shape
    G, N = B.shape[1], A.shape[1]
    d = dd // G
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    u4, dl4, A4 = u.view(b, G, d, L), delta.view(b, G, d, L), A.view(G, d, N)
    h = u.new_zeros(b, G, d, N)
    y = u.new_empty(b, G, d, L)
    tri = None
    for s in range(0, L, T):
        e = min(s + T, L)
        n = e - s
        if tri is None or tri.shape[0] != n:
            tri = torch.ones(n, n, dtype=torch.bool, device=u.device).tril()   # [t, r]: r <= t
        dt = dl4[..., s:e]                                              # b G d n
        S = (dt[..., None] * A4[None, :, :, None, :]).cumsum(3)         # b G d n N
        bu = (dt * u4[..., s:e])[..., None] * B[..., s:e].transpose(-1, -2)[:, :, None]   # b G d n N
        hieu = S[:, :, :, :, None, :] - S[:, :, :, None, :, :]          # b G d t r N
        W = torch.exp(hieu.masked_fill(~tri[:, :, None], float("-inf")))
        x = torch.einsum("bgdtrn,bgdrn->bgdtn", W, bu) + torch.exp(S) * h[:, :, :, None, :]
        y[..., s:e] = torch.einsum("bgdtn,bgtn->bgdt", x, C[..., s:e].transpose(-1, -2))
        h = x[:, :, :, -1]
    y = y.view(b, dd, L)
    return y + u * D.float()[None, :, None] if D is not None else y


def quet_log(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
             return_last_state=False, T=2048, nhom_kenh=96):
    """Bản nhanh: trong mỗi khối T bước, x_t = exp(S_t) * cumsum_r(exp(-S_r) b_r) tính bằng logcumsumexp
    (tách b thành phần dương và âm). Tính ở float64 để S lớn không làm mất độ chính xác. Các nhóm kênh
    độc lập nên tách nhóm để giữ bộ nhớ thấp."""
    assert z is None and not return_last_state
    b, dd, L = u.shape
    G, N = B.shape[1], A.shape[1]
    d = dd // G
    f = torch.float64
    delta = delta.to(f)
    if delta_bias is not None:
        delta = delta + delta_bias.to(f)[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    u4, dl4, A4 = u.to(f).view(b, G, d, L), delta.view(b, G, d, L), A.to(f).view(G, d, N)
    B4, C4 = B.to(f).transpose(-1, -2), C.to(f).transpose(-1, -2)      # b G L N
    y = torch.empty(b, G, d, L, dtype=f, device=u.device)
    for c0 in range(0, d, nhom_kenh):
        c1 = min(c0 + nhom_kenh, d)
        h = torch.zeros(b, G, c1 - c0, N, dtype=f, device=u.device)
        for s in range(0, L, T):
            e = min(s + T, L)
            dt = dl4[:, :, c0:c1, s:e]                                       # b G c n
            S = (dt[..., None] * A4[None, :, c0:c1, None, :]).cumsum(3)     # b G c n N  (<= 0)
            bu = (dt * u4[:, :, c0:c1, s:e])[..., None] * B4[:, :, None, s:e]
            lp = torch.logcumsumexp(torch.log(bu.clamp(min=0)) - S, dim=3)
            ln = torch.logcumsumexp(torch.log((-bu).clamp(min=0)) - S, dim=3)
            x = torch.exp(S + lp) - torch.exp(S + ln) + torch.exp(S) * h[:, :, :, None, :]
            y[:, :, c0:c1, s:e] = (x * C4[:, :, None, s:e]).sum(-1)
            h = x[:, :, :, -1]
    y = y.view(b, dd, L)
    if D is not None:
        y = y + u.to(f) * D.to(f)[None, :, None]
    return y.float()


def quet_log32(u, delta, A, B, C, D=None, z=None, delta_bias=None, delta_softplus=False,
               return_last_state=False, T=256):
    """Như quet_log nhưng float32, trục thời gian để cuối (liền bộ nhớ) cho logcumsumexp nhanh.
    Khối T ngắn để |S| trong khối nhỏ, giữ sai số tương đối khoảng |S| * 1e-7."""
    assert z is None and not return_last_state
    b, dd, L = u.shape
    G, N = B.shape[1], A.shape[1]
    d = dd // G
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    if delta_bias is not None:
        delta = delta + delta_bias.float()[None, :, None]
    if delta_softplus:
        delta = F.softplus(delta)
    u4, dl4, A4 = u.view(b, G, d, L), delta.view(b, G, d, L), A.view(G, d, N)
    y = torch.empty(b, G, d, L, device=u.device)
    h = u.new_zeros(b, G, d, N)
    for s in range(0, L, T):
        e = min(s + T, L)
        dt = dl4[..., s:e]                                                   # b G d n
        S = (dt[:, :, :, None, :] * A4[None, :, :, :, None]).cumsum(-1)     # b G d N n
        bu = (dt * u4[..., s:e])[:, :, :, None, :] * B[:, :, None, :, s:e]   # b G d N n
        lp = torch.logcumsumexp(torch.log(bu.clamp(min=0)) - S, dim=-1)
        ln = torch.logcumsumexp(torch.log((-bu).clamp(min=0)) - S, dim=-1)
        x = torch.exp(S + lp) - torch.exp(S + ln) + torch.exp(S) * h[..., None]
        y[..., s:e] = (x * C[:, :, None, :, s:e]).sum(3)
        h = x[..., -1]
    y = y.view(b, dd, L)
    return y + u * D.float()[None, :, None] if D is not None else y


def tao_vmunet():
    """VMUNet đúng cấu hình notebook huấn luyện (depths [2,2,2,2], decoder [2,2,2,1], 1 lớp, có sigmoid)."""
    if str(VMUNET_REPO) not in sys.path:
        sys.path.insert(0, str(VMUNET_REPO))
    import models.vmunet.vmamba as vm
    from models.vmunet.vmunet import VMUNet
    # quet_log32 (T=256): ~1,2 s/ảnh 640 trên RTX 5050; logit lệch ~1e-4 so với bản float64 (quet_log).
    vm.selective_scan_fn = quet_log32
    return VMUNet(input_channels=3, num_classes=1, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1],
                  drop_path_rate=0.2, load_ckpt_path=None)


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    b, G, d, N, L = 1, 4, 24, 16, 300
    u = torch.randn(b, G * d, L, device=dev)
    delta = torch.randn(b, G * d, L, device=dev)
    A = -torch.exp(torch.randn(G * d, N, device=dev)).clamp(max=16)
    Bm, Cm = torch.randn(b, G, N, L, device=dev), torch.randn(b, G, N, L, device=dev)
    D, bias = torch.randn(G * d, device=dev), torch.randn(G * d, device=dev)
    kw = dict(delta_bias=bias, delta_softplus=True)
    ref = quet_tuan_tu(u, delta, A, Bm, Cm, D, **kw)
    for T in (1, 7, 64, 512):
        out = quet_theo_khoi(u, delta, A, Bm, Cm, D, T=T, **kw)
        err = ((out - ref).abs().max() / ref.abs().max()).item()
        print(f"T={T:3d}: sai số tương đối lớn nhất {err:.2e}")
        assert err < 1e-4
    print("OK: bản theo khối khớp bản tuần tự")
