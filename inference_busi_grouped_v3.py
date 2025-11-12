# =============================================================
# multiverseg_busi_finetune_v4.py
# Full‑layer fine‑tune of MultiverSeg on BUSI with CLAHE + Cosine scheduler
# - Domain‑aware preprocessing (Z‑score, gamma, log, CLAHE, median despeckle)
# - Grouped support by class (benign/malignant/normal) with per‑class prototypes
# - Full retrain (all layers unfrozen) with BCE+Dice loss
# - CosineAnnealingLR + optional AMP
# - Saves best checkpoint by val Dice
# =============================================================
import os
import cv2
import math
import time
import copy
import json
import random
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image, ImageFilter
from tqdm import tqdm

# ======== Config ========
class CFG:
    ROOT = r"C:\\limu\\limu007\\limu007\\Dataset_BUSI_with_GT"
    OUTDIR = r"C:\\limu\\limu007\\limu007\\runs_busi_v4"
    MVS_WEIGHT = r"C:\\anaconda\\envs\\multiverseg\\Lib\\site-packages\\checkpoints\\MultiverSeg_v1_nf256_res128.pt"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = 256

    BATCH = 6
    EPOCHS = 30               # you can raise to 50 for better results
    LR = 1e-4
    WEIGHT_DECAY = 1e-5
    NUM_WORKERS = 0
    AMP = True                # mixed precision

    # Support prototypes
    SUPPORT_PER_CLASS = 24    # number of images per class used to compute prototype (averaged)
    CLASSES = ['benign', 'malignant', 'normal']

    # Augmentations
    HFLIP_P = 0.5
    VFLIP_P = 0.2
    ROT90_P = 0.2
    BRIGHT_JITTER = 0.08      # +/- range (multiplicative)
    CONTRAST_JITTER = 0.10

    # Scheduler
    SCHED_TMAX = EPOCHS
    MIN_LR = 5e-6

    # Save / log
    SAVE_EVERY = 5

os.makedirs(CFG.OUTDIR, exist_ok=True)

# ======== Ultrasound domain preprocessing ========
def zscore(img):
    m, s = img.mean(), img.std()
    return (img - m) / (s + 1e-8)

def gamma_adjust(img, gamma=0.9):
    return np.clip(img ** gamma, 0, 1)

def log_compress(img, scale=5.0):
    return np.log1p(scale * img) / np.log1p(scale)

def clahe_enhance(img, clip=2.0, tile=(8,8)):
    # img in [0,1]
    cl = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
    x = (img * 255).astype(np.uint8)
    x = cl.apply(x)
    return x.astype(np.float32) / 255.0

def despeckle_median(img, k=3):
    pil = Image.fromarray((img * 255).astype(np.uint8))
    filt = pil.filter(ImageFilter.MedianFilter(size=k))
    return np.array(filt, dtype=np.float32) / 255.0

# Simple jitter
def jitter(img):
    # multiplicative brightness & contrast (light)
    b = 1.0 + random.uniform(-CFG.BRIGHT_JITTER, CFG.BRIGHT_JITTER)
    c = 1.0 + random.uniform(-CFG.CONTRAST_JITTER, CFG.CONTRAST_JITTER)
    mu = img.mean()
    out = (img - mu) * c + mu
    out = np.clip(out * b, 0, 1)
    return out

# geometric augs
def geo_augs(img, msk):
    if random.random() < CFG.HFLIP_P:
        img = np.ascontiguousarray(np.flip(img, axis=1))
        msk = np.ascontiguousarray(np.flip(msk, axis=1))
    if random.random() < CFG.VFLIP_P:
        img = np.ascontiguousarray(np.flip(img, axis=0))
        msk = np.ascontiguousarray(np.flip(msk, axis=0))
    if random.random() < CFG.ROT90_P:
        k = random.choice([1,2,3])
        img = np.ascontiguousarray(np.rot90(img, k))
        msk = np.ascontiguousarray(np.rot90(msk, k))
    return img, msk

# Full pipeline
def ultrasound_preprocess(img, train=False):
    img = zscore(img)
    img = gamma_adjust(img, 0.9)
    img = log_compress(img, 5.0)
    img = clahe_enhance(img, 2.0, (8,8))
    img = despeckle_median(img, 3)
    if train:
        img = jitter(img)
    img = np.clip(img, 0, 1)
    return img

# ======== Dataset ========
class BUSIDataset(Dataset):
    def __init__(self, root: str, indices: list, labels: list, train: bool):
        self.root = Path(root)
        self.indices = indices
        self.labels = labels  # class string per index
        self.train = train
        # materialize file lists
        self.img_files, self.msk_files = [], []
        all_samples = []
        for cls in CFG.CLASSES:
            img_dir = self.root/cls/'images'
            msk_dir = self.root/cls/'mask'
            for f in os.listdir(img_dir):
                if f.endswith('.png'):
                    m = f.replace('.png','_mask.png')
                    if (msk_dir/m).exists():
                        all_samples.append((str(img_dir/f), str(msk_dir/m), cls))
        for i in self.indices:
            self.img_files.append(all_samples[i][0])
            self.msk_files.append(all_samples[i][1])

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, i):
        img = Image.open(self.img_files[i]).convert('L').resize((CFG.IMG_SIZE, CFG.IMG_SIZE))
        msk = Image.open(self.msk_files[i]).convert('L').resize((CFG.IMG_SIZE, CFG.IMG_SIZE))
        img = np.array(img, np.float32)/255.0
        msk = (np.array(msk, np.float32) > 0.5).astype(np.float32)
        img = ultrasound_preprocess(img, train=self.train)
        if self.train:
            img, msk = geo_augs(img, msk)
        img_t = torch.from_numpy(img).unsqueeze(0)  # [1,H,W]
        msk_t = torch.from_numpy(msk).unsqueeze(0)
        return img_t, msk_t

# ======== Index split with labels ========
def build_index_split(root: str, test_size=0.3, seed=42):
    all_samples, labels = [], []
    rootp = Path(root)
    for cls in CFG.CLASSES:
        img_dir = rootp/cls/'images'
        msk_dir = rootp/cls/'mask'
        for f in os.listdir(img_dir):
            if not f.endswith('.png'): continue
            m = f.replace('.png','_mask.png')
            if (msk_dir/m).exists():
                all_samples.append((str(img_dir/f), str(msk_dir/m), cls))
                labels.append(cls)
    idxs = list(range(len(all_samples)))
    tr, te = train_test_split(idxs, test_size=test_size, random_state=seed, stratify=labels)
    # also make a small held‑out val from train
    tr_labels = [labels[i] for i in tr]
    tr_idx, val_idx = train_test_split(tr, test_size=0.1, random_state=seed, stratify=tr_labels)
    return tr_idx, val_idx, te, labels

# ======== MultiverSeg wrapper ========
from multiverseg.models.sp_mvs import MultiverSegNet

class WrappedMVS(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.net = MultiverSegNet(
            in_channels=[5, 2],
            encoder_blocks=[256,256,256,256],
            block_kws=dict(conv_kws=dict(norm='layer')),
            cross_relu=True
        ).to(device)

    def forward(self, q5, s_img, s_msk):
        """
        q5:     [B, 5, H, W] → will be expanded to [B, 1, 5, H, W]
        s_img:  [1, 1, H, W] or [B, 1, H, W]
        s_msk:  [1, 1, H, W] or [B, 1, H, W]
        """
        B, _, H, W = q5.shape
        q5 = q5.unsqueeze(1)  # [B, 1, 5, H, W]

        if s_img.ndim == 4:
            s_img = s_img.unsqueeze(1)
        if s_msk.ndim == 4:
            s_msk = s_msk.unsqueeze(1)

        # broadcast support if needed
        if s_img.shape[0] == 1 and B > 1:
            s_img = s_img.expand(B, -1, -1, -1, -1)
            s_msk = s_msk.expand(B, -1, -1, -1, -1)

        return self.net(q5, s_img, s_msk)

    @torch.no_grad()
    def predict(self, q1, proto_img, proto_msk):
        """
        q1:         [B, 1, H, W] (grayscale input)
        proto_img:  [1, 1, H, W] or [B, 1, H, W]
        proto_msk:  [1, 1, H, W] or [B, 1, H, W]
        """
        q5 = q1.repeat(1, 5, 1, 1)         # 擴成 5 通道
        logits = self.forward(q5, proto_img, proto_msk)
        return torch.sigmoid(logits)        # 回傳 0~1 概率

# ======== Prototype builder (per class averages) ========
@torch.no_grad()
def build_class_prototypes(loader: DataLoader, per_class=CFG.SUPPORT_PER_CLASS, device=CFG.DEVICE):
    buckets: Dict[str, list] = {c: [] for c in CFG.CLASSES}
    # one pass over loader until we collect enough per class
    for imgs, msks in loader:
        B = imgs.size(0)
        # crude heuristic: infer class from file order is unavailable here; sample uniformly
        # so we just push into round‑robin buckets
        for b in range(B):
            # rotate class assignment to keep balance
            c = CFG.CLASSES[(len(buckets['benign'])+len(buckets['malignant'])+len(buckets['normal'])) % 3]
            if len(buckets[c]) < per_class:
                buckets[c].append((imgs[b:b+1].to(device), msks[b:b+1].to(device)))
        if all(len(buckets[c]) >= per_class for c in CFG.CLASSES):
            break
    protos = {}
    for c in CFG.CLASSES:
        if len(buckets[c]) == 0:
            # fallback neutral prototype
            proto_img = torch.zeros(1,1,CFG.IMG_SIZE, CFG.IMG_SIZE, device=device) + 0.5
            proto_msk = torch.zeros_like(proto_img)
        else:
            imgs = torch.cat([x for x,_ in buckets[c]], dim=0)  # [N,1,H,W]
            msks = torch.cat([y for _,y in buckets[c]], dim=0)
            proto_img = imgs.mean(dim=0, keepdim=True)         # [1,1,H,W]
            proto_msk = msks.mean(dim=0, keepdim=True)
        protos[c] = (proto_img, proto_msk)
    return protos

# ======== Loss ========
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        num = 2 * (probs*targets).sum(dim=(1,2,3))
        den = (probs+targets).sum(dim=(1,2,3)) + self.eps
        dice = 1 - (num/den)
        return dice.mean()

# ======== Train / Val loops ========

def compute_metrics_bin(y_pred, y_true):
    eps = 1e-8
    y_pred = (y_pred>0.5).int().flatten()
    y_true = y_true.int().flatten()
    TP = ((y_pred==1)&(y_true==1)).sum().item()
    TN = ((y_pred==0)&(y_true==0)).sum().item()
    FP = ((y_pred==1)&(y_true==0)).sum().item()
    FN = ((y_pred==0)&(y_true==1)).sum().item()
    acc  = (TP+TN)/(TP+TN+FP+FN+eps)
    prec = TP/(TP+FP+eps)
    rec  = TP/(TP+FN+eps)
    iou  = TP/(TP+FP+FN+eps)
    dice = 2*TP/(2*TP+FP+FN+eps)
    return acc,prec,rec,iou,dice


def run():
    device = torch.device(CFG.DEVICE)

    # Build index split
    tr_idx, val_idx, te_idx, labels = build_index_split(CFG.ROOT, test_size=0.3, seed=42)

    # Datasets
    tr_ds = BUSIDataset(CFG.ROOT, tr_idx, labels, train=True)
    val_ds = BUSIDataset(CFG.ROOT, val_idx, labels, train=False)
    te_ds  = BUSIDataset(CFG.ROOT, te_idx, labels, train=False)

    tr_loader = DataLoader(tr_ds, batch_size=CFG.BATCH, shuffle=True, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CFG.BATCH, shuffle=False, num_workers=CFG.NUM_WORKERS)
    te_loader  = DataLoader(te_ds,  batch_size=1, shuffle=False, num_workers=CFG.NUM_WORKERS)

    # Model
    model = WrappedMVS(device)
    # Load initial weights (optional, speeds convergence)
    if os.path.exists(CFG.MVS_WEIGHT):
        state = torch.load(CFG.MVS_WEIGHT, map_location=device)
        if 'model' in state:
            model.net.load_state_dict(state['model'])
        else:
            model.load_state_dict(state)
        print('[INFO] Loaded initial MultiverSeg weights')
    model = model.to(device)

    # Optim, sched
    opt = torch.optim.AdamW(model.parameters(), lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.SCHED_TMAX, eta_min=CFG.MIN_LR)

    bce = nn.BCEWithLogitsLoss()
    dice = DiceLoss()

    scaler = torch.cuda.amp.GradScaler(enabled=CFG.AMP and device.type=='cuda')

    # Build prototypes (per class) from a subset of train loader
    proto_loader = DataLoader(tr_ds, batch_size=CFG.BATCH, shuffle=True, num_workers=0)
    prototypes = build_class_prototypes(proto_loader, per_class=CFG.SUPPORT_PER_CLASS, device=device)

    # Helper to choose prototype (cycle through classes to avoid bias)
    proto_cycle = {c:prototypes[c] for c in CFG.CLASSES}

    best_dice = -1
    best_path = os.path.join(CFG.OUTDIR, 'best_mvs.pt')

    # ========== Train ==========
    for epoch in range(1, CFG.EPOCHS+1):
        model.train()
        tloss = 0.0
        pbar = tqdm(tr_loader, desc=f'[Train] epoch {epoch}/{CFG.EPOCHS}')
        for imgs, msks in pbar:
            imgs = imgs.to(device)
            msks = msks.to(device)
            # pick a prototype by cycling classes
            # here we blend all classes equally
            proto_img = torch.cat([proto_cycle[c][0] for c in CFG.CLASSES], dim=0).mean(dim=0, keepdim=True)
            proto_msk = torch.cat([proto_cycle[c][1] for c in CFG.CLASSES], dim=0).mean(dim=0, keepdim=True)

            q5 = imgs.repeat(1,5,1,1)
            with torch.cuda.amp.autocast(enabled=CFG.AMP and device.type=='cuda'):
                logits = model.forward(q5, proto_img, proto_msk)
                loss = 0.5*bce(logits, msks) + 0.5*dice(logits, msks)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            tloss += loss.item()*imgs.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}")
        tloss /= len(tr_loader.dataset)

        # Validate
        model.eval()
        md = []
        with torch.no_grad():
            for imgs, msks in DataLoader(val_ds, batch_size=1, shuffle=False):
                imgs = imgs.to(device); msks = msks.to(device)
                y = model.predict(imgs, proto_img, proto_msk)
                acc,prec,rec,iou,dc = compute_metrics_bin(y.cpu(), msks.cpu())
                md.append(dc)
        mean_dice = float(np.mean(md)) if md else 0.0
        sched.step()

        # Save
        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save({'model': model.state_dict(), 'dice': best_dice, 'epoch': epoch}, best_path)
        if epoch % CFG.SAVE_EVERY == 0:
            torch.save({'model': model.state_dict(), 'epoch': epoch}, os.path.join(CFG.OUTDIR, f'ckpt_e{epoch}.pt'))

        print(f"[E{epoch}] train_loss={tloss:.4f}  val_dice={mean_dice:.4f}  lr={sched.get_last_lr()[0]:.2e}")

    print(f"[INFO] Best val Dice: {best_dice:.4f}  -> {best_path}")

    # ========== Test (load best) ==========
    if os.path.exists(best_path):
        ck = torch.load(best_path, map_location=device)
        model.load_state_dict(ck['model'])

    metrics = {'acc': [], 'prec': [], 'rec': [], 'iou': [], 'dice': []}
    with torch.no_grad():
        for imgs, msks in tqdm(te_loader, desc='[Test]'):
            imgs = imgs.to(device); msks = msks.to(device)
            y = model.predict(imgs, proto_img, proto_msk)
            a,p,r,i,d = compute_metrics_bin(y.cpu(), msks.cpu())
            for k,v in zip(metrics.keys(), [a,p,r,i,d]):
                metrics[k].append(v)

    print("\n=== BUSI v4 Test Results ===")
    for k,v in metrics.items():
        print(f"{k.upper():<8}: {np.mean(v):.4f} ± {np.std(v):.4f}")


if __name__ == '__main__':
    run()
