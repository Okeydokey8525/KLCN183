# -*- coding: utf-8 -*-
"""Sinh hai notebook Kaggle: Polyp-PVT và VM-UNet (BG20, 640, 5 seed).

Sửa nội dung ở file này rồi chạy lại để sinh notebook, không sửa tay trong .ipynb:
    python tao_notebook.py
Các phần dùng chung (cấu hình, dữ liệu, vòng huấn luyện) chép từ notebook PraNet 640 của nhóm,
nên hai mô hình mới dùng đúng giao thức dữ liệu, tăng cường và đánh giá như PraNet/U-Net 640.
"""
import json
import os

THU_MUC = os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------------
# CELL 0: cấu hình (khác nhau ở MODEL_NAME, CHUAN_HOA)
# ----------------------------------------------------------------------------
CELL0 = r'''# ============================================================
# CELL 0: CẤU HÌNH LƯỢT CHẠY (CHỈ SỬA SEED VÀ CHAY_THU)
# ============================================================
# Chỉ dùng 1 GPU. Phải đặt TRƯỚC khi import torch.
import os, glob
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = "__MODEL__"
SEED = 0             # chạy lần lượt 0, 1, 2, 3, 4 (đã chốt trước, không thêm seed sau khi xem kết quả)
CHAY_THU = False     # True: chỉ chạy 2 epoch để đo thời gian/epoch và bộ nhớ GPU

IMG_SIZE = 640       # giống YOLO và PraNet/U-Net 640
EPOCHS = 2 if CHAY_THU else 100
CLOSE_MOSAIC = 1 if CHAY_THU else 10   # tắt mosaic ở 10 epoch cuối, giống YOLO
BATCH_SIZE = 8       # thiếu bộ nhớ GPU thì đổi BATCH_SIZE = 4, ACCUM = 2 (vẫn tương đương batch 8)
ACCUM = 1
GIOI_HAN_GIO = 11.0  # Kaggle cắt phiên ở 12 giờ: quá mốc này thì lưu resume.pth rồi dừng để còn giữ được kết quả
CHUAN_HOA = "imagenet"   # cả hai backbone đều tiền huấn luyện trên ImageNet

# Tăng cường lấy đúng giá trị trong args.yaml của 40 lượt YOLO (giống PraNet/U-Net 640)
AUG = dict(hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, translate=0.1, scale=0.5, fliplr=0.5, mosaic=1.0)
assert SEED in range(5), "Hướng 1 chỉ chạy seed 0-4"


def tim(mau):
    return sorted(glob.glob(mau, recursive=True))


TREN_KAGGLE = os.path.isdir("/kaggle/input")
GOC_INPUT = "/kaggle/input" if TREN_KAGGLE else os.environ.get("THU_MUC_INPUT", ".")
WORK_DIR = "/kaggle/working" if TREN_KAGGLE else os.environ.get("THU_MUC_LAM_VIEC", os.getcwd())

_bg20 = tim(f"{GOC_INPUT}/**/Kvasir_YOLO_SEG_BG20/images/train")
assert _bg20, "❌ Chưa thêm dataset BG20 (kvasir-yolo-seg-bg20) vào notebook"
DATASET_ROOT = os.path.dirname(os.path.dirname(_bg20[0]))

RUN_NAME = f"{MODEL_NAME}_640_aug_s{SEED}" + ("_chaythu" if CHAY_THU else "")
OUTPUT_DIR = f"{WORK_DIR}/runs/{RUN_NAME}"
print(f"{MODEL_NAME}: SEED={SEED}, IMG_SIZE={IMG_SIZE}, EPOCHS={EPOCHS}, batch {BATCH_SIZE} x {ACCUM}")
print("Dữ liệu:", DATASET_ROOT)

import torch
assert torch.cuda.is_available(), "❌ Chưa bật GPU: Settings > Accelerator > GPU T4"
print(f"GPU: {torch.cuda.get_device_name(0)}, torch {torch.__version__}")
'''

# ----------------------------------------------------------------------------
# CELL 1: dữ liệu BG20 + tăng cường giống YOLO (chép nguyên từ pranet-new.ipynb)
# ----------------------------------------------------------------------------
CELL1 = r'''# ============================================================
# CELL 1: DATASET BG20 + TĂNG CƯỜNG GIỐNG YOLO (MOSAIC, CO GIÃN, DỊCH, HSV, LẬT NGANG)
# ============================================================
# Chép nguyên từ notebook PraNet 640 của nhóm để cả 4 mô hình so sánh dùng cùng một cách nạp dữ liệu.
#  - Ảnh được kéo về IMG_SIZE x IMG_SIZE (không letterbox).
#  - Mosaic 4 ảnh, sau đó affine (co giãn ±50%, dịch ±10%) cắt về IMG_SIZE, giống random_perspective
#    với degrees = shear = perspective = 0 như cấu hình YOLO.
#  - Không dùng erasing: Ultralytics chỉ dùng erasing cho bài phân loại.
import os, glob, random, cv2, torch
import numpy as np
from torch.utils.data import Dataset


def doc_anh_mat_na(img_path, label_dir):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    stem = os.path.splitext(os.path.basename(img_path))[0]
    label_path = os.path.join(label_dir, stem + ".txt")
    if os.path.exists(label_path) and os.path.getsize(label_path) > 0:
        with open(label_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) > 5:  # đa giác YOLO-seg
                    coords = np.array([float(x) for x in parts[1:]], dtype=np.float32).reshape(-1, 2)
                    coords[:, 0] *= w
                    coords[:, 1] *= h
                    cv2.fillPoly(mask, [coords.astype(np.int32)], 255)
    return img, mask


def hsv_ngau_nhien(img, hgain, sgain, vgain):
    """Giống RandomHSV của Ultralytics."""
    r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1
    hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_RGB2HSV))
    x = np.arange(0, 256, dtype=r.dtype)
    lut_hue = ((x * r[0]) % 180).astype(np.uint8)
    lut_sat = np.clip(x * r[1], 0, 255).astype(np.uint8)
    lut_val = np.clip(x * r[2], 0, 255).astype(np.uint8)
    hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def affine_ngau_nhien(img, mask, out, scale, translate):
    """Co giãn và dịch quanh tâm ảnh, cắt về out x out (random_perspective với góc xoay = 0)."""
    h, w = img.shape[:2]
    C = np.eye(3); C[0, 2], C[1, 2] = -w / 2, -h / 2
    s = random.uniform(1 - scale, 1 + scale)
    S = np.eye(3); S[0, 0] = S[1, 1] = s
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * out
    M = (T @ S @ C)[:2]
    img = cv2.warpAffine(img, M, (out, out), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))
    mask = cv2.warpAffine(mask, M, (out, out), flags=cv2.INTER_NEAREST, borderValue=0)
    return img, mask


class PolypBG20(Dataset):
    def __init__(self, root_dir, split, img_size, augment, chuan_hoa):
        self.img_dir = os.path.join(root_dir, "images", split)
        self.label_dir = os.path.join(root_dir, "labels", split)
        self.img_files = sorted(glob.glob(os.path.join(self.img_dir, "*.jpg")))
        assert len(self.img_files) > 0, f"Không tìm thấy ảnh tại: {self.img_dir}"
        self.s, self.augment, self.chuan_hoa = img_size, augment, chuan_hoa
        self.mosaic = augment and AUG["mosaic"] > 0   # tắt ở CLOSE_MOSAIC epoch cuối

    def __len__(self):
        return len(self.img_files)

    def anh_vuong(self, i):
        img, mask = doc_anh_mat_na(self.img_files[i], self.label_dir)
        img = cv2.resize(img, (self.s, self.s), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.s, self.s), interpolation=cv2.INTER_NEAREST)
        return img, mask

    def mosaic4(self, i):
        """Mosaic 4 ảnh giống Ultralytics: tâm ngẫu nhiên trên khung 2s x 2s rồi affine cắt về s x s."""
        s = self.s
        ids = [i] + random.choices(range(len(self)), k=3)
        yc, xc = (int(random.uniform(s // 2, 3 * s // 2)) for _ in range(2))
        khung = np.full((2 * s, 2 * s, 3), 114, dtype=np.uint8)
        kmask = np.zeros((2 * s, 2 * s), dtype=np.uint8)
        for k, j in enumerate(ids):
            img, mask = self.anh_vuong(j)
            h, w = img.shape[:2]
            if k == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h
            elif k == 1:
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, 2 * s), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif k == 2:
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
            else:
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, 2 * s), min(2 * s, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            khung[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]
            kmask[y1a:y2a, x1a:x2a] = mask[y1b:y2b, x1b:x2b]
        return affine_ngau_nhien(khung, kmask, s, AUG["scale"], AUG["translate"])

    def __getitem__(self, i):
        if self.augment:
            if self.mosaic and random.random() < AUG["mosaic"]:
                img, mask = self.mosaic4(i)
            else:
                img, mask = affine_ngau_nhien(*self.anh_vuong(i), self.s, AUG["scale"], AUG["translate"])
            img = hsv_ngau_nhien(img, AUG["hsv_h"], AUG["hsv_s"], AUG["hsv_v"])
            if random.random() < AUG["fliplr"]:
                img, mask = np.ascontiguousarray(img[:, ::-1]), np.ascontiguousarray(mask[:, ::-1])
        else:
            img, mask = self.anh_vuong(i)

        img = img.astype(np.float32) / 255.0
        if self.chuan_hoa == "imagenet":
            img = (img - np.array([0.485, 0.456, 0.406], np.float32)) / np.array([0.229, 0.224, 0.225], np.float32)
        img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
        mask_tensor = torch.tensor(mask > 127, dtype=torch.float32).unsqueeze(0)
        return img_tensor, mask_tensor


def khoi_tao_worker(worker_id):
    """Mỗi worker có seed riêng nhưng lặp lại được theo SEED."""
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws)
    random.seed(ws)


train_ds = PolypBG20(DATASET_ROOT, "train", IMG_SIZE, augment=True, chuan_hoa=CHUAN_HOA)
val_ds = PolypBG20(DATASET_ROOT, "val", IMG_SIZE, augment=False, chuan_hoa=CHUAN_HOA)
print(f"Tập train: {len(train_ds)} ảnh (880 polyp + 160 nền), có tăng cường")
print(f"Tập val  : {len(val_ds)} ảnh (120 polyp + 40 nền), không tăng cường")
assert len(train_ds) == 1040 and len(val_ds) == 160, "Sai số ảnh BG20"

_x, _m = train_ds[0]
assert _x.shape == (3, IMG_SIZE, IMG_SIZE) and _m.shape == (1, IMG_SIZE, IMG_SIZE)
assert set(torch.unique(_m).tolist()) <= {0.0, 1.0}
print("✅ Dataset và tăng cường hoạt động đúng")
'''

# ----------------------------------------------------------------------------
# CELL 2a: Polyp-PVT
# ----------------------------------------------------------------------------
CELL2_PVT = r'''# ============================================================
# CELL 2: MÔ HÌNH POLYP-PVT (MÃ GỐC CỦA TÁC GIẢ) + HÀM MẤT MÁT
# ============================================================
# Mã nguồn: https://github.com/DengPingFan/Polyp-PVT (lib/pvt.py, lib/pvtv2.py), nạp từ dataset Kaggle,
# không chép lại kiến trúc. Giấy phép của repo: chỉ dùng cho nghiên cứu và giáo dục.
# Thiết lập theo Train.py gốc: AdamW lr 1e-4, weight_decay 1e-4, cắt gradient theo từng phần tử ±0,5,
# loss = structure_loss(P1) + structure_loss(P2), dự đoán = sigmoid(P1 + P2).
# lr giữ 1e-4 suốt quá trình: adjust_lr gốc chỉ giảm từ epoch 200, mà bản gốc chạy 99 epoch.
# Khác bản gốc (để cùng giao thức với PraNet/U-Net 640): ảnh 640, tăng cường giống YOLO,
# không huấn luyện đa tỉ lệ {0,75; 1; 1,25}, batch 8 thay vì 16, 100 epoch.
import sys, shutil, functools
import torch
import torch.nn as nn
import torch.nn.functional as F

_pvt = tim(f"{GOC_INPUT}/**/lib/pvt.py")
assert _pvt, "❌ Chưa thêm dataset chứa mã Polyp-PVT (thư mục có lib/pvt.py)"
REPO_DIR = os.path.dirname(os.path.dirname(_pvt[0]))
_w = tim(f"{GOC_INPUT}/**/pvt_v2_b2.pth")
assert _w, "❌ Chưa có pvt_v2_b2.pth trong dataset"
TRONG_SO = _w[0]
print("Mã Polyp-PVT:", REPO_DIR)
print("Trọng số PVTv2-B2:", TRONG_SO)

# lib/pvt.py nạp cứng './pretrained_pth/pvt_v2_b2.pth' theo thư mục hiện hành
os.makedirs(f"{WORK_DIR}/pretrained_pth", exist_ok=True)
_dich = f"{WORK_DIR}/pretrained_pth/pvt_v2_b2.pth"
if not os.path.exists(_dich):
    try:
        os.symlink(TRONG_SO, _dich)
    except OSError:
        shutil.copy(TRONG_SO, _dich)
os.chdir(WORK_DIR)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from lib.pvt import PolypPVT

CKPT_NAME = "polyp_pvt_best_s{}.pth"
LR, WEIGHT_DECAY, GRAD_CLIP = 1e-4, 1e-4, 0.5


def tao_mo_hinh():
    # torch >= 2.6 mặc định weights_only=True; file trọng số tải từ link của tác giả nên cho nạp đầy đủ
    _goc = torch.load
    torch.load = functools.partial(_goc, weights_only=False, map_location="cpu")
    try:
        m = PolypPVT()
    finally:
        torch.load = _goc
    # Kiểm tra backbone đã nhận trọng số ImageNet
    sd = _goc(TRONG_SO, map_location="cpu", weights_only=False)
    bb = m.backbone.state_dict()
    khop = sum(1 for k, v in sd.items() if k in bb and bb[k].shape == v.shape and torch.equal(bb[k], v))
    print(f"Backbone nhận {khop}/{len(bb)} tensor từ pvt_v2_b2.pth")
    assert khop >= 0.9 * len(bb), "❌ Backbone không nạp được trọng số tiền huấn luyện"
    return m


def tao_toi_uu(model):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    return opt, None   # lr không đổi, giống bản gốc trong 99 epoch


def structure_loss(pred, mask):
    """Weighted BCE + weighted IoU, như Train.py của Polyp-PVT (và PraNet gốc)."""
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction="none")
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def tinh_loss(out, masks):
    p1, p2 = out
    return structure_loss(p1, masks) + structure_loss(p2, masks)


def xac_suat(out):
    p1, p2 = out
    return torch.sigmoid(p1 + p2)


def phien_ban():
    import timm
    return f"torch={torch.__version__} timm={timm.__version__}"
'''

# ----------------------------------------------------------------------------
# CELL 2b: VM-UNet
# ----------------------------------------------------------------------------
CELL2_VMUNET = r'''# ============================================================
# CELL 2: MÔ HÌNH VM-UNET (MÃ GỐC CỦA TÁC GIẢ) + HÀM MẤT MÁT
# ============================================================
# Mã nguồn: https://github.com/JCruan519/VM-UNet (models/vmunet/), giấy phép Apache-2.0, nạp từ dataset Kaggle.
# Cần kernel CUDA selective_scan của gói mamba_ssm.
# Thiết lập theo configs/config_setting.py gốc: depths [2,2,2,2], depths_decoder [2,2,2,1], drop_path 0,2,
# trọng số VMamba-S (vmamba_small_e238_ema.pth) cho encoder và decoder (giống load_from gốc),
# BceDiceLoss (wb=1, wd=1), AdamW lr 1e-3, betas (0,9; 0,999), eps 1e-8, weight_decay 1e-2,
# CosineAnnealingLR T_max 50, eta_min 1e-5, ngưỡng 0,5.
# Khác bản gốc (để cùng giao thức với PraNet/U-Net 640): ảnh 640 thay 256, tăng cường giống YOLO
# (bản gốc lật dọc, xoay), chuẩn hóa ImageNet, batch 8 thay 32, 100 epoch thay 300.
import sys, importlib.util, subprocess
import torch
import torch.nn as nn


def cai_mamba():
    if importlib.util.find_spec("mamba_ssm") is not None:
        return
    whl = tim(f"{GOC_INPUT}/**/mamba_ssm-*.whl")
    if whl:
        print("Cài mamba_ssm từ wheel:", whl[0])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-deps", whl[0]])
    else:
        print("Không thấy wheel mamba_ssm trong dataset, cài từ PyPI (cần bật Internet; có thể phải biên dịch 30-60 phút)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "mamba-ssm", "--no-build-isolation"])


cai_mamba()
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
# Thử kernel trên GPU trước khi chạy thật
_u = torch.randn(1, 4, 16, device="cuda")
_y = selective_scan_fn(_u, torch.rand(1, 4, 16, device="cuda"), -torch.rand(4, 2, device="cuda"),
                       torch.randn(1, 1, 2, 16, device="cuda"), torch.randn(1, 1, 2, 16, device="cuda"),
                       torch.ones(4, device="cuda"), delta_softplus=True)
assert torch.isfinite(_y).all()
print("✅ Kernel selective_scan chạy được")

_v = tim(f"{GOC_INPUT}/**/models/vmunet/vmunet.py")
assert _v, "❌ Chưa thêm dataset chứa mã VM-UNet (thư mục có models/vmunet/vmunet.py)"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(_v[0])))
_w = tim(f"{GOC_INPUT}/**/vmamba_small_e238_ema.pth")
assert _w, "❌ Chưa có vmamba_small_e238_ema.pth trong dataset"
TRONG_SO = _w[0]
print("Mã VM-UNet:", REPO_DIR)
print("Trọng số VMamba-S:", TRONG_SO)
if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)
from models.vmunet.vmunet import VMUNet

CKPT_NAME = "vmunet_best_s{}.pth"
GRAD_CLIP = None


def nap_trong_so(model, path):
    """Giống VMUNet.load_from gốc: encoder nhận layers.i, decoder nhận layers_up.(3-i).
    Viết lại ở đây chỉ để nạp được trên torch mới (weights_only) và bỏ tensor lệch kích thước."""
    # File checkpoint VMamba có lưu kèm cấu hình kiểu yacs.CfgNode; Kaggle không cài yacs nên tạo lớp thay thế
    try:
        import yacs.config  # noqa: F401
    except ImportError:
        import types
        _y, _yc = types.ModuleType("yacs"), types.ModuleType("yacs.config")
        _yc.CfgNode = type("CfgNode", (dict,), {})
        _y.config = _yc
        sys.modules["yacs"], sys.modules["yacs.config"] = _y, _yc
    ck = torch.load(path, map_location="cpu", weights_only=False)["model"]
    msd = model.vmunet.state_dict()
    enc = {k: v for k, v in ck.items() if k in msd and msd[k].shape == v.shape}
    doi = (("layers.0", "layers_up.3"), ("layers.1", "layers_up.2"),
           ("layers.2", "layers_up.1"), ("layers.3", "layers_up.0"))
    dec = {}
    for k, v in ck.items():
        for a, b in doi:
            if a in k:
                dec[k.replace(a, b)] = v
                break
    dec = {k: v for k, v in dec.items() if k in msd and msd[k].shape == v.shape}
    msd.update(enc)
    msd.update(dec)
    model.vmunet.load_state_dict(msd)
    print(f"Encoder nhận {len(enc)} tensor, decoder nhận {len(dec)} tensor; mô hình có {len(msd)} tensor")
    assert len(enc) > 0.5 * len([k for k in msd if k.startswith(("patch_embed", "layers."))]), \
        "❌ Encoder không nạp được trọng số tiền huấn luyện"


def tao_mo_hinh():
    m = VMUNet(input_channels=3, num_classes=1, depths=[2, 2, 2, 2], depths_decoder=[2, 2, 2, 1],
               drop_path_rate=0.2, load_ckpt_path=None)
    nap_trong_so(m, TRONG_SO)
    return m


def tao_toi_uu(model):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50, eta_min=1e-5)
    return opt, sch


_bce = nn.BCELoss()


def tinh_loss(out, masks):
    """BceDiceLoss(wb=1, wd=1) của repo gốc. Đầu ra VMUNet đã qua sigmoid (num_classes = 1)."""
    b = out.size(0)
    p, t = out.view(b, -1), masks.view(b, -1)
    dice = 1 - ((2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)).sum() / b
    return _bce(p, t) + dice


def xac_suat(out):
    return out


def phien_ban():
    import timm, mamba_ssm
    return f"torch={torch.__version__} timm={timm.__version__} mamba_ssm={mamba_ssm.__version__}"
'''

# ----------------------------------------------------------------------------
# CELL 3: huấn luyện (dùng chung)
# ----------------------------------------------------------------------------
CELL3 = r'''# ============================================================
# CELL 3: KHÓA TẤT ĐỊNH VÀ HUẤN LUYỆN (640, CÓ TĂNG CƯỜNG, ĐÁNH GIÁ VAL MỖI EPOCH)
# ============================================================
import os, csv, time, random, shutil
import numpy as np
import torch
from torch.utils.data import DataLoader

os.environ["PYTHONHASHSEED"] = str(SEED)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

DEVICE = torch.device("cuda:0")
os.makedirs(OUTPUT_DIR, exist_ok=True)

g = torch.Generator()
g.manual_seed(SEED)
SO_WORKER = 2 if TREN_KAGGLE else 0
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=SO_WORKER, pin_memory=True,
                          worker_init_fn=khoi_tao_worker, generator=g)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=SO_WORKER, pin_memory=True)

model = tao_mo_hinh().to(DEVICE)
optimizer, scheduler = tao_toi_uu(model)
print(f"Số tham số: {sum(p.numel() for p in model.parameters()):,}")

csv_path = os.path.join(OUTPUT_DIR, "results.csv")
best_val_dice, best_epoch, epoch_dau, thoi_gian_truoc = -1.0, 0, 1, 0.0

# Chạy tiếp từ lượt bị dừng ở mốc GIOI_HAN_GIO (thêm output của lượt cũ làm input của notebook)
_resume = tim(f"{GOC_INPUT}/**/{RUN_NAME}/resume.pth")
if _resume:
    ck = torch.load(_resume[0], map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    if scheduler is not None and ck["scheduler"] is not None:
        scheduler.load_state_dict(ck["scheduler"])
    best_val_dice, best_epoch, epoch_dau, thoi_gian_truoc = ck["best_val_dice"], ck["best_epoch"], ck["epoch"] + 1, ck["time_s"]
    torch.set_rng_state(ck["rng_cpu"]); torch.cuda.set_rng_state(ck["rng_cuda"])
    np.random.set_state(ck["rng_np"]); random.setstate(ck["rng_py"]); g.set_state(ck["rng_loader"])
    thu_muc_cu = os.path.dirname(_resume[0])
    for ten in ["results.csv", CKPT_NAME.format(SEED)]:
        shutil.copy(os.path.join(thu_muc_cu, ten), os.path.join(OUTPUT_DIR, ten))
    print(f"↻ Chạy tiếp từ epoch {epoch_dau} (best Dice {best_val_dice:.4f} ở epoch {best_epoch})")
else:
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_dice", "val_iou", "lr", "mosaic", "time_s", "gpu_mem_gb"])

print(f"Bắt đầu huấn luyện {MODEL_NAME}: SEED={SEED}, IMG_SIZE={IMG_SIZE}, EPOCHS={EPOCHS}")
t0 = time.time() - thoi_gian_truoc
t_phien = time.time()
xong = True

for epoch in range(epoch_dau, EPOCHS + 1):
    t_ep = time.time()
    train_ds.mosaic = AUG["mosaic"] > 0 and epoch <= EPOCHS - CLOSE_MOSAIC
    torch.cuda.reset_peak_memory_stats()

    model.train()
    total_loss = 0.0
    optimizer.zero_grad()
    n_batch = len(train_loader)
    for b, (imgs, masks) in enumerate(train_loader, 1):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        loss = tinh_loss(model(imgs), masks)
        (loss / ACCUM).backward()
        if b % ACCUM == 0 or b == n_batch:
            if GRAD_CLIP is not None:   # Polyp-PVT: cắt từng phần tử gradient như clip_gradient gốc
                for p in model.parameters():
                    if p.grad is not None:
                        p.grad.data.clamp_(-GRAD_CLIP, GRAD_CLIP)
            optimizer.step()
            optimizer.zero_grad()
        total_loss += loss.item() * imgs.size(0)
    lr_now = optimizer.param_groups[0]["lr"]
    if scheduler is not None:
        scheduler.step()
    train_loss = total_loss / len(train_ds)

    # Val mỗi epoch: Dice, IoU theo từng ảnh; ảnh nền tính đúng nếu không báo điểm nào (giống PraNet 640)
    model.eval()
    dices, ious = [], []
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            probs = xac_suat(model(imgs)) > 0.5
            for p, gt in zip(probs, masks > 0.5):
                inter = (p & gt).sum().item()
                union = (p | gt).sum().item()
                tong = p.sum().item() + gt.sum().item()
                if gt.sum().item() == 0:
                    d = iou = 1.0 if p.sum().item() == 0 else 0.0
                else:
                    d, iou = 2.0 * inter / (tong + 1e-5), inter / (union + 1e-5)
                dices.append(d)
                ious.append(iou)
    val_dice, val_iou = float(np.mean(dices)), float(np.mean(ious))
    elapsed = time.time() - t0
    mem = torch.cuda.max_memory_allocated() / 1024**3

    with open(csv_path, "a", newline="") as f:
        csv.writer(f).writerow([epoch, f"{train_loss:.6f}", f"{val_dice:.6f}", f"{val_iou:.6f}",
                                f"{lr_now:.8f}", int(train_ds.mosaic), f"{elapsed:.1f}", f"{mem:.2f}"])

    mark = ""
    if val_dice > best_val_dice:
        best_val_dice, best_epoch = val_dice, epoch
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, CKPT_NAME.format(SEED)))
        mark = " (BEST SAVED)"
    t_mot_epoch = time.time() - t_ep
    if epoch % 5 == 0 or epoch == epoch_dau or mark:
        print(f"Epoch [{epoch:03d}/{EPOCHS:03d}] loss {train_loss:.4f} | Val Dice {val_dice:.4f} | "
              f"Val IoU {val_iou:.4f} | {elapsed / 60:.1f} phút | GPU {mem:.1f} GB{mark}")
    if epoch == epoch_dau:
        print(f"   ≈ {t_mot_epoch / 60:.1f} phút/epoch → cả {EPOCHS} epoch ≈ {t_mot_epoch * EPOCHS / 3600:.1f} giờ")

    # Sắp chạm giới hạn phiên Kaggle: lưu trạng thái để chạy tiếp ở phiên sau
    if epoch < EPOCHS and (time.time() - t_phien) + 1.2 * t_mot_epoch > GIOI_HAN_GIO * 3600:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "epoch": epoch, "best_val_dice": best_val_dice, "best_epoch": best_epoch,
                    "time_s": time.time() - t0, "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state(), "rng_np": np.random.get_state(),
                    "rng_py": random.getstate(), "rng_loader": g.get_state()},
                   os.path.join(OUTPUT_DIR, "resume.pth"))
        print(f"⏸ Dừng ở epoch {epoch} vì gần hết {GIOI_HAN_GIO} giờ. Đã lưu resume.pth: "
              f"thêm output này làm input rồi chạy lại notebook với cùng SEED.")
        xong = False
        break

if xong:
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "last.pth"))
    if os.path.exists(os.path.join(OUTPUT_DIR, "resume.pth")):
        os.remove(os.path.join(OUTPUT_DIR, "resume.pth"))
with open(os.path.join(OUTPUT_DIR, "tom_tat.txt"), "w") as f:
    f.write(f"model={MODEL_NAME} seed={SEED} img_size={IMG_SIZE} epochs={EPOCHS} batch={BATCH_SIZE}x{ACCUM} aug={AUG}\n")
    f.write(f"best_epoch={best_epoch} best_val_dice={best_val_dice:.6f} train_time_s={time.time() - t0:.1f} "
            f"xong={xong} resumed={bool(_resume)}\n")
    f.write(f"{phien_ban()} gpu={torch.cuda.get_device_name(0)}\n")
print(f"\n{'Xong' if xong else 'Tạm dừng'}. Val Dice tốt nhất {best_val_dice:.4f} ở epoch {best_epoch}; "
      f"thời gian {(time.time() - t0) / 3600:.2f} giờ")
'''

CELL4 = r'''# ============================================================
# CELL 4: NÉN KẾT QUẢ ĐỂ TẢI VỀ (tab Output)
# ============================================================
!cd /kaggle/working && zip -r "runs_{RUN_NAME}.zip" runs
'''

CELL_WHEEL = r'''# ============================================================
# CHẠY MỘT LẦN: ĐÓNG GÓI WHEEL mamba_ssm CHO KAGGLE (cần bật GPU và Internet)
# ============================================================
# Kết quả: /kaggle/working/wheels/mamba_ssm-*.whl. Tải về rồi thêm vào dataset vm-unet-code,
# để 6 tài khoản cài lại trong vài giây, cùng một phiên bản, không phải biên dịch lại.
import subprocess, sys, glob, torch
print("torch", torch.__version__, "CUDA", torch.version.cuda, "Python", sys.version.split()[0])
subprocess.check_call([sys.executable, "-m", "pip", "wheel", "mamba-ssm", "--no-deps",
                       "--no-build-isolation", "-w", "/kaggle/working/wheels"])
whl = glob.glob("/kaggle/working/wheels/mamba_ssm-*.whl")
print(whl)
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-deps", whl[0]])
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
u = torch.randn(1, 4, 16, device="cuda")
y = selective_scan_fn(u, torch.rand(1, 4, 16, device="cuda"), -torch.rand(4, 2, device="cuda"),
                      torch.randn(1, 1, 2, 16, device="cuda"), torch.randn(1, 1, 2, 16, device="cuda"),
                      torch.ones(4, device="cuda"), delta_softplus=True)
print("✅ Kernel chạy được:", y.shape)
'''


def notebook(cells, internet):
    return {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                   "source": c.splitlines(keepends=True)} for c in cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "gpu", "isGpuEnabled": True, "isInternetEnabled": internet,
                       "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4, "nbformat_minor": 4,
    }


def ghi(ten, nb):
    duong_dan = os.path.join(THU_MUC, "notebooks", ten)
    with open(duong_dan, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print("Đã ghi", duong_dan)


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    os.makedirs(os.path.join(THU_MUC, "notebooks"), exist_ok=True)
    ghi("polyp-pvt-640.ipynb", notebook([CELL0.replace("__MODEL__", "PolypPVT"), CELL1, CELL2_PVT, CELL3, CELL4], False))
    ghi("vm-unet-640.ipynb", notebook([CELL0.replace("__MODEL__", "VMUNet"), CELL1, CELL2_VMUNET, CELL3, CELL4], True))
    ghi("dong-goi-mamba-wheel.ipynb", notebook([CELL_WHEEL], True))
