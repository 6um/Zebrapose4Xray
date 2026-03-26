import os
import glob
import random
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
# 1. 随机种子
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
    数据目录结构假设为：

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

        # xray: (H, W) uint8 -> float32 [0,1]
        xray = Image.open(xray_path).convert("L")
        xray = np.array(xray, dtype=np.float32) / 255.0

        # mask: (H, W), 0/1
        mask = np.load(mask_path).astype(np.float32)

        # code_stack: (10, H, W), 0/1
        code_stack = np.load(code_path).astype(np.float32)

        # 输入 shape: (1, H, W)
        x = np.expand_dims(xray, axis=0)

        # target shape: (11, H, W)
        y = np.concatenate([mask[None, ...], code_stack], axis=0)

        x = torch.from_numpy(x)       # float32
        y = torch.from_numpy(y)       # float32

        return x, y


# ============================================================
# 3. ResNet Encoder + Decoder
# ============================================================

class ResNetSegmentation(nn.Module):
    """
    用 ResNet18 做 encoder，再用简单上采样 decoder 做像素级预测。
    输出 11 通道：
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

        # 把第一层改成单通道输入
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
        self.maxpool = base.maxpool   # /4
        self.layer1 = base.layer1     # /4
        self.layer2 = base.layer2     # /8
        self.layer3 = base.layer3     # /16
        self.layer4 = base.layer4     # /32

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

        # Decoder
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

    # mask IoU
    inter = (mask_pred * mask_gt).sum(dim=(1, 2, 3))
    union = ((mask_pred + mask_gt) > 0).float().sum(dim=(1, 2, 3))
    mask_iou = ((inter + 1e-6) / (union + 1e-6)).mean().item()

    # code bit accuracy（全部像素平均）
    code_acc = (code_pred == code_gt).float().mean().item()

    # 前景区域内的 code accuracy
    fg = mask_gt.expand_as(code_gt)
    fg_count = fg.sum()
    if fg_count > 0:
        fg_code_acc = ((code_pred == code_gt).float() * fg).sum() / (fg_count + 1e-6)
        fg_code_acc = fg_code_acc.item()
    else:
        fg_code_acc = 0.0

    return {
        "mask_iou": mask_iou,
        "code_acc": code_acc,
        "fg_code_acc": fg_code_acc,
    }


# ============================================================
# 6. Train / Val
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
    mean_metrics = {
        "mask_iou": np.mean([m["mask_iou"] for m in metric_list]),
        "code_acc": np.mean([m["code_acc"] for m in metric_list]),
        "fg_code_acc": np.mean([m["fg_code_acc"] for m in metric_list]),
    }

    return {
        "loss": total_loss / n,
        "mask_loss": total_mask_loss / n,
        "code_loss": total_code_loss / n,
        **mean_metrics
    }


# ============================================================
# 7. Main
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = ResNetSegmentation(
        out_channels=11,
        backbone=args.backbone,
        pretrained=args.pretrained
    ).to(device)

    criterion = MultiTaskLoss(mask_weight=2.0, code_weight=1.0)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_log = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_log = validate_one_epoch(model, val_loader, criterion, device)

        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_log['loss']:.4f} "
            f"train_mask={train_log['mask_loss']:.4f} "
            f"train_code={train_log['code_loss']:.4f} | "
            f"val_loss={val_log['loss']:.4f} "
            f"val_mask={val_log['mask_loss']:.4f} "
            f"val_code={val_log['code_loss']:.4f} "
            f"val_iou={val_log['mask_iou']:.4f} "
            f"val_code_acc={val_log['code_acc']:.4f} "
            f"val_fg_code_acc={val_log['fg_code_acc']:.4f}"
        )

        # 保存 last
        last_ckpt = os.path.join(args.save_dir, "last.pth")
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_log["loss"],
                "args": vars(args),
            },
            last_ckpt,
        )

        # 保存 best
        if val_log["loss"] < best_val_loss:
            best_val_loss = val_log["loss"]
            best_ckpt = os.path.join(args.save_dir, "best.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_log["loss"],
                    "args": vars(args),
                },
                best_ckpt,
            )
            print(f"[INFO] Best model saved to {best_ckpt}")


if __name__ == "__main__":
    main()

'''
python train.py \
    --data_root ./dataset \
    --save_dir ./checkpoints \
    --epochs 50 \
    --batch_size 2 \
    --lr 1e-3 \
    --backbone resnet34


'''