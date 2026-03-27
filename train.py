import os
import glob
import random
import json
import csv
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models


# ============================================================
# 1. Random Seed
# ============================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# 2. Dataset
# ============================================================

class XrayCodeDataset(Dataset):
    """
    Expected dataset directory structure:

    dataset/
        sample_000000/
            xray.png
            mask.npy
            code_stack.npy
            sample_meta.json
        sample_000001/
            ...
    """

    def __init__(self, root_dir: str):
        self.root_dir = root_dir
        self.sample_dirs = sorted(
            [p for p in glob.glob(os.path.join(root_dir, "sample_*")) if os.path.isdir(p)]
        )

        if len(self.sample_dirs) == 0:
            raise RuntimeError(f"No sample_* folders found in {root_dir}")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]

        xray_path = os.path.join(sample_dir, "xray.png")
        mask_path = os.path.join(sample_dir, "mask.npy")
        code_path = os.path.join(sample_dir, "code_stack.npy")

        xray = Image.open(xray_path).convert("L")
        xray = np.array(xray, dtype=np.float32) / 255.0

        mask = np.load(mask_path).astype(np.float32)
        code_stack = np.load(code_path).astype(np.float32)

        x = np.expand_dims(xray, axis=0)
        y = np.concatenate([mask[None, ...], code_stack], axis=0)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, y


# ============================================================
# 3. ResNet Encoder + Decoder
# ============================================================

class ResNetSegmentation(nn.Module):
    """
    ResNet encoder + simple decoder for pixel-wise prediction.
    Output channels:
        0: mask
        1~10: code bits
    """
    def __init__(self, out_channels: int = 11, backbone: str = "resnet18", pretrained: bool = False):
        super().__init__()

        if backbone == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            base = models.resnet18(weights=weights)
            feat_channels = [64, 64, 128, 256, 512]
        elif backbone == "resnet34":
            weights = models.ResNet34_Weights.DEFAULT if pretrained else None
            base = models.resnet34(weights=weights)
            feat_channels = [64, 64, 128, 256, 512]
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        old_conv1 = base.conv1
        base.conv1 = nn.Conv2d(
            1, old_conv1.out_channels,
            kernel_size=old_conv1.kernel_size,
            stride=old_conv1.stride,
            padding=old_conv1.padding,
            bias=False
        )

        if pretrained:
            with torch.no_grad():
                base.conv1.weight[:] = old_conv1.weight.mean(dim=1, keepdim=True)

        self.stem = nn.Sequential(
            base.conv1,   # /2
            base.bn1,
            base.relu
        )
        self.maxpool = base.maxpool  # downsample /4
        self.layer1 = base.layer1  # /4
        self.layer2 = base.layer2  # /8
        self.layer3 = base.layer3  # /16
        self.layer4 = base.layer4  # /32

        # Decoder
        self.up4 = self._up_block(feat_channels[4], 256)
        self.up3 = self._up_block(256 + feat_channels[3], 128)
        self.up2 = self._up_block(128 + feat_channels[2], 64)
        self.up1 = self._up_block(64 + feat_channels[1], 64)
        self.up0 = self._up_block(64 + feat_channels[0], 32)

        self.head = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=1)
        )

    def _up_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        x0 = self.stem(x)         # [B, 64, H/2,  W/2]
        x1 = self.maxpool(x0)     # [B, 64, H/4,  W/4]
        x1 = self.layer1(x1)      # [B, 64, H/4,  W/4]
        x2 = self.layer2(x1)      # [B,128, H/8,  W/8]
        x3 = self.layer3(x2)      # [B,256, H/16, W/16]
        x4 = self.layer4(x3)      # [B,512, H/32, W/32]

        # Decoder with skip connections
        d4 = F.interpolate(x4, size=x3.shape[-2:], mode="bilinear", align_corners=False)
        d4 = self.up4(d4)

        d3 = torch.cat([d4, x3], dim=1)
        d3 = F.interpolate(d3, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.up3(d3)

        d2 = torch.cat([d3, x2], dim=1)
        d2 = F.interpolate(d2, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.up2(d2)

        d1 = torch.cat([d2, x1], dim=1)
        d1 = F.interpolate(d1, size=x0.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.up1(d1)

        d0 = torch.cat([d1, x0], dim=1)
        d0 = F.interpolate(d0, scale_factor=2, mode="bilinear", align_corners=False)
        d0 = self.up0(d0)

        out = self.head(d0)  # [B, 11, H, W]
        return out


# ============================================================
# 4. Loss
# ============================================================

class MultiTaskLoss(nn.Module):
    def __init__(self, mask_weight: float = 2.0, code_weight: float = 1.0):
        super().__init__()
        self.mask_weight = mask_weight
        self.code_weight = code_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        """
        logits:  [B, 11, H, W]
        targets: [B, 11, H, W]
        """
        mask_logits = logits[:, 0:1]
        code_logits = logits[:, 1:]

        mask_targets = targets[:, 0:1]
        code_targets = targets[:, 1:]

        mask_loss = self.bce(mask_logits, mask_targets)
        code_loss = self.bce(code_logits, code_targets)

        total_loss = self.mask_weight * mask_loss + self.code_weight * code_loss
        return total_loss, mask_loss.detach(), code_loss.detach()


# ============================================================
# 5. Metrics
# ============================================================

@torch.no_grad()
def compute_metrics(logits, targets, threshold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    mask_pred = preds[:, 0:1]
    mask_gt = targets[:, 0:1]

    code_pred = preds[:, 1:]
    code_gt = targets[:, 1:]

    eps = 1e-6

    # -------------------------
    # Mask metrics
    # -------------------------
    inter = (mask_pred * mask_gt).sum(dim=(1, 2, 3))
    union = ((mask_pred + mask_gt) > 0).float().sum(dim=(1, 2, 3))
    # mask_iou:
    # Intersection over Union (IoU) for the predicted mask.
    # Measures the overlap between predicted foreground and ground-truth foreground.
    # Value range: [0, 1], higher is better.
    mask_iou = ((inter + eps) / (union + eps)).mean().item()

    tp = ((mask_pred == 1) & (mask_gt == 1)).float().sum(dim=(1, 2, 3))
    fp = ((mask_pred == 1) & (mask_gt == 0)).float().sum(dim=(1, 2, 3))
    fn = ((mask_pred == 0) & (mask_gt == 1)).float().sum(dim=(1, 2, 3))
    tn = ((mask_pred == 0) & (mask_gt == 0)).float().sum(dim=(1, 2, 3))

    # mask_precision:
    # Among all pixels predicted as foreground, how many are truly foreground.
    # High precision means few false positive foreground pixels.
    mask_precision = ((tp + eps) / (tp + fp + eps)).mean().item()

    # mask_recall:
    # Among all true foreground pixels, how many are successfully detected.
    # High recall means few missed foreground pixels.
    mask_recall = ((tp + eps) / (tp + fn + eps)).mean().item()

    # mask_f1:
    # Harmonic mean of mask precision and mask recall.
    # Useful when both false positives and false negatives matter.
    mask_f1 = ((2 * tp + eps) / (2 * tp + fp + fn + eps)).mean().item()

    # mask_acc:
    # Overall mask pixel accuracy over the whole image.
    # Can be high even when foreground prediction is imperfect,
    # especially if background occupies most pixels.
    mask_acc = ((tp + tn + eps) / (tp + tn + fp + fn + eps)).mean().item()

    # -------------------------
    # Code metrics
    # -------------------------

    # code_acc:
    # Bit-wise accuracy over all code channels and all pixels.
    # This includes both foreground and background.
    # Note: this metric can be inflated by easy background pixels,
    # because background code values are usually all zero.
    code_acc = (code_pred == code_gt).float().mean().item()

    # fg:
    # Foreground mask expanded to all code channels.
    # Used to evaluate code prediction only inside object regions.
    fg = mask_gt.expand_as(code_gt)

    # bg:
    # Background region expanded to all code channels.
    bg = 1.0 - fg

    fg_count = fg.sum()
    bg_count = bg.sum()

    # fg_code_acc:
    # Bit-wise code accuracy computed only inside foreground pixels.
    # This is usually more meaningful than code_acc for your task,
    # because binary codes are only semantically important on the object.
    if fg_count > 0:
        fg_code_acc = (((code_pred == code_gt).float() * fg).sum() / (fg_count + eps)).item()
    else:
        fg_code_acc = 0.0

    # bg_code_acc:
    # Bit-wise code accuracy computed only on background pixels.
    # Usually expected to be high because background code is typically zero.
    # Mostly useful as a sanity check rather than a main evaluation metric.
    if bg_count > 0:
        bg_code_acc = (((code_pred == code_gt).float() * bg).sum() / (bg_count + eps)).item()
    else:
        bg_code_acc = 0.0

    # bg_code_acc:
    # Bit-wise code accuracy computed only on background pixels.
    # Usually expected to be high because background code is typically zero.
    # Mostly useful as a sanity check rather than a main evaluation metric.
    code_tp_fg = (((code_pred == 1) & (code_gt == 1)).float() * fg).sum()
    code_fp_fg = (((code_pred == 1) & (code_gt == 0)).float() * fg).sum()
    code_fn_fg = (((code_pred == 0) & (code_gt == 1)).float() * fg).sum()

    # code_bitwise_precision_fg:
    # Among all predicted positive code bits in the foreground,
    # how many are actually correct positive bits.
    code_bitwise_precision_fg = ((code_tp_fg + eps) / (code_tp_fg + code_fp_fg + eps)).item()

    # code_bitwise_recall_fg:
    # Among all true positive code bits in the foreground,
    # how many are successfully predicted.
    code_bitwise_recall_fg = ((code_tp_fg + eps) / (code_tp_fg + code_fn_fg + eps)).item()

    # code_bitwise_f1_fg:
    # Harmonic mean of foreground bit-wise precision and recall.
    # Gives a balanced summary of foreground code prediction quality.
    code_bitwise_f1_fg = ((2 * code_tp_fg + eps) / (2 * code_tp_fg + code_fp_fg + code_fn_fg + eps)).item()

    # Foreground full-code correctness:
    # for each foreground pixel, all 10 bits must be correct

    # bit_equal:
    # Per-bit correctness map for the 10 code channels.
    # Shape: [B, 10, H, W], value 1 means the bit is predicted correctly.
    bit_equal = (code_pred == code_gt).float()

    # all_bits_correct_per_pixel:
    # Per-pixel strict correctness for the whole binary code.
    # A pixel is counted as correct only if all 10 bits are correct simultaneously.
    # This is much stricter than fg_code_acc.
    all_bits_correct_per_pixel = (bit_equal.mean(dim=1, keepdim=True) == 1.0).float()

    # fg_pixel_mask:
    # Foreground mask at pixel level, shape [B, 1, H, W].
    fg_pixel_mask = mask_gt

    fg_pixel_count = fg_pixel_mask.sum()

    # code_fg_all_correct:
    # Foreground full-code accuracy.
    # Measures the fraction of foreground pixels whose entire 10-bit code
    # is predicted perfectly.
    # This is one of the strictest and most informative metrics
    # for structured binary code prediction.
    if fg_pixel_count > 0:
        code_fg_all_correct = ((all_bits_correct_per_pixel * fg_pixel_mask).sum() / (fg_pixel_count + eps)).item()
    else:
        code_fg_all_correct = 0.0

    return {
        "mask_iou": mask_iou,
        "mask_precision": mask_precision,
        "mask_recall": mask_recall,
        "mask_f1": mask_f1,
        "mask_acc": mask_acc,
        "code_acc": code_acc,
        "fg_code_acc": fg_code_acc,
        "bg_code_acc": bg_code_acc,
        "code_bitwise_precision_fg": code_bitwise_precision_fg,
        "code_bitwise_recall_fg": code_bitwise_recall_fg,
        "code_bitwise_f1_fg": code_bitwise_f1_fg,
        "code_fg_all_correct": code_fg_all_correct,
    }


# ============================================================
# 6. History saving
# ============================================================

def save_history_json(history, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, ensure_ascii=False)


def save_history_csv(history, save_path):
    if len(history) == 0:
        return

    fieldnames = list(history[0].keys())
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)


# ============================================================
# 7. Train / Val
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()

    total_loss = 0.0
    total_mask_loss = 0.0
    total_code_loss = 0.0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss, mask_loss, code_loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_mask_loss += mask_loss.item()
        total_code_loss += code_loss.item()

    n = len(loader)
    return {
        "loss": total_loss / n,
        "mask_loss": total_mask_loss / n,
        "code_loss": total_code_loss / n,
    }


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_mask_loss = 0.0
    total_code_loss = 0.0

    metric_list = []

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss, mask_loss, code_loss = criterion(logits, y)

        total_loss += loss.item()
        total_mask_loss += mask_loss.item()
        total_code_loss += code_loss.item()

        metric_list.append(compute_metrics(logits, y))

    n = len(loader)

    mean_metrics = {}
    if len(metric_list) > 0:
        for k in metric_list[0].keys():
            mean_metrics[k] = float(np.mean([m[k] for m in metric_list]))

    return {
        "loss": total_loss / n,
        "mask_loss": total_mask_loss / n,
        "code_loss": total_code_loss / n,
        **mean_metrics
    }


# ============================================================
# 8. Main
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True, help="dataset root")
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--backbone", type=str, default="resnet18", choices=["resnet18", "resnet34"])
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"[INFO] device = {device}")

    dataset = XrayCodeDataset(args.data_root)
    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_ratio))
    n_train = n_total - n_val

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed)
    )

    pin_memory = torch.cuda.is_available()

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    model = ResNetSegmentation(
        out_channels=11,
        backbone=args.backbone,
        pretrained=args.pretrained
    ).to(device)

    criterion = MultiTaskLoss(mask_weight=2.0, code_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")
    best_epoch = -1

    history = []
    history_json_path = os.path.join(args.save_dir, "history.json")
    history_csv_path = os.path.join(args.save_dir, "history.csv")
    best_info_path = os.path.join(args.save_dir, "best_metrics.json")

    for epoch in range(1, args.epochs + 1):
        train_log = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_log = validate_one_epoch(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]

        record = {
            "epoch": epoch,
            "lr": float(current_lr),

            "train_loss": float(train_log["loss"]),
            "train_mask_loss": float(train_log["mask_loss"]),
            "train_code_loss": float(train_log["code_loss"]),

            "val_loss": float(val_log["loss"]),
            "val_mask_loss": float(val_log["mask_loss"]),
            "val_code_loss": float(val_log["code_loss"]),

            "val_mask_iou": float(val_log.get("mask_iou", 0.0)),
            "val_mask_precision": float(val_log.get("mask_precision", 0.0)),
            "val_mask_recall": float(val_log.get("mask_recall", 0.0)),
            "val_mask_f1": float(val_log.get("mask_f1", 0.0)),
            "val_mask_acc": float(val_log.get("mask_acc", 0.0)),

            "val_code_acc": float(val_log.get("code_acc", 0.0)),
            "val_fg_code_acc": float(val_log.get("fg_code_acc", 0.0)),
            "val_bg_code_acc": float(val_log.get("bg_code_acc", 0.0)),
            "val_code_bitwise_precision_fg": float(val_log.get("code_bitwise_precision_fg", 0.0)),
            "val_code_bitwise_recall_fg": float(val_log.get("code_bitwise_recall_fg", 0.0)),
            "val_code_bitwise_f1_fg": float(val_log.get("code_bitwise_f1_fg", 0.0)),
            "val_code_fg_all_correct": float(val_log.get("code_fg_all_correct", 0.0)),
        }

        history.append(record)
        save_history_json(history, history_json_path)
        save_history_csv(history, history_csv_path)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_log['loss']:.4f} "
            f"train_mask={train_log['mask_loss']:.4f} "
            f"train_code={train_log['code_loss']:.4f} | "
            f"val_loss={val_log['loss']:.4f} "
            f"val_mask={val_log['mask_loss']:.4f} "
            f"val_code={val_log['code_loss']:.4f} "
            f"val_iou={val_log.get('mask_iou', 0.0):.4f} "
            f"val_code_acc={val_log.get('code_acc', 0.0):.4f} "
            f"val_fg_code_acc={val_log.get('fg_code_acc', 0.0):.4f} "
            f"val_code_fg_all_correct={val_log.get('code_fg_all_correct', 0.0):.4f}"
        )

        last_ckpt = os.path.join(args.save_dir, "last.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_log["loss"],
                "args": vars(args),
                "history": history,
            },
            last_ckpt,
        )

        if val_log["loss"] < best_val_loss:
            best_val_loss = val_log["loss"]
            best_epoch = epoch

            best_ckpt = os.path.join(args.save_dir, "best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_log["loss"],
                    "args": vars(args),
                    "history": history,
                },
                best_ckpt,
            )

            best_info = {
                "best_epoch": best_epoch,
                "best_val_loss": float(best_val_loss),
                "metrics_at_best": record,
            }
            with open(best_info_path, "w", encoding="utf-8") as f:
                json.dump(best_info, f, indent=2, ensure_ascii=False)

            print(f"[INFO] Best model saved to {best_ckpt}")

    print(f"[INFO] Training finished. Best epoch = {best_epoch}, best val_loss = {best_val_loss:.6f}")
    print(f"[INFO] History saved to:")
    print(f"       {history_json_path}")
    print(f"       {history_csv_path}")
    print(f"       {best_info_path}")


if __name__ == "__main__":
    main()




'''
python train.py \
    --data_root ./dataset_test \
    --save_dir ./checkpoints \
    --epochs 50 \
    --batch_size 8 \
    --lr 1e-3 \
    --backbone resnet34

'''

