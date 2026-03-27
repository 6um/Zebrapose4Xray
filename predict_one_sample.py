import os
import json
import argparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


# ============================================================
# 1. Model
# ============================================================

class ResNetSegmentation(nn.Module):
    """
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
            base.conv1,
            base.bn1,
            base.relu
        )
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4

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
        x0 = self.stem(x)
        x1 = self.maxpool(x0)
        x1 = self.layer1(x1)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

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

        return self.head(d0)


# ============================================================
# 2. Basic utils
# ============================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_gray_png(arr: np.ndarray, path: str):
    Image.fromarray(arr.astype(np.uint8), mode="L").save(path)


def save_rgb_png(arr: np.ndarray, path: str):
    Image.fromarray(arr.astype(np.uint8), mode="RGB").save(path)


def load_xray_image(sample_dir: str):
    xray_path = os.path.join(sample_dir, "xray.png")
    if not os.path.exists(xray_path):
        raise FileNotFoundError(f"xray.png not found in {sample_dir}")

    xray = Image.open(xray_path).convert("L")
    xray_np = np.array(xray, dtype=np.float32) / 255.0

    x = np.expand_dims(xray_np, axis=0)   # (1, H, W)
    x = np.expand_dims(x, axis=0)         # (1, 1, H, W)

    return torch.from_numpy(x), (xray_np * 255.0).astype(np.uint8)


def load_optional_gt(sample_dir: str):
    gt_mask = None
    gt_code_stack = None

    mask_npy = os.path.join(sample_dir, "mask.npy")
    code_npy = os.path.join(sample_dir, "code_stack.npy")

    if os.path.exists(mask_npy):
        gt_mask = np.load(mask_npy).astype(np.uint8)

    if os.path.exists(code_npy):
        gt_code_stack = np.load(code_npy).astype(np.uint8)

    return gt_mask, gt_code_stack


def pack_code_stack_to_uint16(code_stack: np.ndarray) -> np.ndarray:
    if code_stack.ndim != 3 or code_stack.shape[0] != 10:
        raise ValueError("code_stack must have shape (10, H, W)")

    packed = np.zeros(code_stack.shape[1:], dtype=np.uint16)
    for i in range(10):
        packed |= (code_stack[i].astype(np.uint16) << i)
    return packed


def make_code_visualization(packed_code: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vis = (packed_code.astype(np.float32) / 1023.0 * 255.0).clip(0, 255).astype(np.uint8)
    vis[mask == 0] = 0
    return vis


def make_mask_overlay(xray_gray_u8: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgb = np.stack([xray_gray_u8, xray_gray_u8, xray_gray_u8], axis=-1).astype(np.uint8)
    overlay = rgb.copy()
    overlay[mask == 1, 0] = 255
    overlay[mask == 1, 1] = (overlay[mask == 1, 1] * 0.4).astype(np.uint8)
    overlay[mask == 1, 2] = (overlay[mask == 1, 2] * 0.4).astype(np.uint8)
    return overlay


# ============================================================
# 3. Bit-level visualization
# ============================================================

def bit_to_black_white_image(bit_img: np.ndarray, mask: np.ndarray, bg_gray: int = 127) -> np.ndarray:
    """
    Visualization rule:
        foreground bit = 0 -> black
        foreground bit = 1 -> white
        background      -> gray
    """
    bit_img = bit_img.astype(np.uint8)
    mask = mask.astype(np.uint8)

    vis = np.full(bit_img.shape, bg_gray, dtype=np.uint8)   # background = gray
    vis[(mask == 1) & (bit_img == 1)] = 255                 # foreground 1 = white
    vis[(mask == 1) & (bit_img == 0)] = 0                   # foreground 0 = black
    return vis


def save_bit_images(code_stack: np.ndarray, mask: np.ndarray, out_dir: str, prefix: str):
    os.makedirs(out_dir, exist_ok=True)
    saved_paths = []

    for level in range(code_stack.shape[0]):
        bit = code_stack[level]
        vis = bit_to_black_white_image(bit, mask)
        out_path = os.path.join(out_dir, f"{prefix}_bit_{level+1:02d}.png")
        save_gray_png(vis, out_path)
        saved_paths.append(out_path)

    return saved_paths

def get_default_font(size=22):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_labeled_tile_from_gray(gray_img: np.ndarray, title: str, title_h: int = 30, border: int = 2) -> Image.Image:
    img = Image.fromarray(gray_img.astype(np.uint8), mode="L").convert("RGB")
    w, h = img.size

    canvas = Image.new("RGB", (w + 2 * border, h + title_h + 2 * border), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = get_default_font(size=18)

    draw.rectangle([0, 0, canvas.size[0] - 1, canvas.size[1] - 1], outline=(0, 0, 0), width=1)

    try:
        bbox = draw.textbbox((0, 0), title, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
    except Exception:
        tw = int(draw.textlength(title, font=font))
        th = 16

    tx = (canvas.size[0] - tw) // 2
    ty = max(2, (title_h - th) // 2)
    draw.text((tx, ty), title, fill=(0, 0, 0), font=font)

    canvas.paste(img, (border, title_h + border))
    return canvas

def draw_labeled_tile_from_bit(
    bit_img: np.ndarray,
    mask: np.ndarray,
    title: str,
    title_h: int = 30,
    border: int = 2
) -> Image.Image:
    vis = bit_to_black_white_image(bit_img, mask)
    return draw_labeled_tile_from_gray(vis, title, title_h=title_h, border=border)


def create_2x10_bit_grid(
    gt_code_stack: np.ndarray,
    gt_mask: np.ndarray,
    pred_code_stack: np.ndarray,
    pred_mask: np.ndarray,
    save_path: str,
    gap: int = 8,
    title_h: int = 30,
    border: int = 2
):
    if gt_code_stack is None:
        raise ValueError("GT code_stack is required.")
    if pred_code_stack is None:
        raise ValueError("Prediction code_stack is required.")
    if gt_code_stack.shape[0] != 10 or pred_code_stack.shape[0] != 10:
        raise ValueError("Both GT and pred code stacks must have 10 levels.")

    gt_tiles = [
        draw_labeled_tile_from_bit(gt_code_stack[i], gt_mask, f"GT L{i + 1}", title_h=title_h, border=border)
        for i in range(10)
    ]
    pred_tiles = [
        draw_labeled_tile_from_bit(pred_code_stack[i], pred_mask, f"Pred L{i + 1}", title_h=title_h, border=border)
        for i in range(10)
    ]

    tile_w, tile_h = gt_tiles[0].size
    canvas_w = 10 * tile_w + 9 * gap
    canvas_h = 2 * tile_h + gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    for i, tile in enumerate(gt_tiles):
        canvas.paste(tile, (i * (tile_w + gap), 0))

    for i, tile in enumerate(pred_tiles):
        canvas.paste(tile, (i * (tile_w + gap), tile_h + gap))

    canvas.save(save_path)


# ============================================================
# 4. Panel visualization
# ============================================================

def make_panel_2x2(
    gt_mask: np.ndarray,
    gt_code_vis: np.ndarray,
    pred_mask: np.ndarray,
    pred_code_vis: np.ndarray,
    save_path: str,
    title_h: int = 30,
    gap: int = 10,
    border: int = 2
):
    """
    2x2 panel:

    Row 1: GT Mask   | GT Code
    Row 2: Pred Mask | Pred Code
    """
    # infer image size from prediction
    h, w = pred_mask.shape
    blank = np.full((h, w), 255, dtype=np.uint8)

    gt_mask_img = gt_mask * 255 if gt_mask is not None else blank
    gt_code_img = gt_code_vis if gt_code_vis is not None else blank
    pred_mask_img = pred_mask * 255
    pred_code_img = pred_code_vis

    tiles = [
        draw_labeled_tile_from_gray(gt_mask_img, "GT Mask", title_h=title_h, border=border),
        draw_labeled_tile_from_gray(gt_code_img, "GT Code", title_h=title_h, border=border),
        draw_labeled_tile_from_gray(pred_mask_img, "Pred Mask", title_h=title_h, border=border),
        draw_labeled_tile_from_gray(pred_code_img, "Pred Code", title_h=title_h, border=border),
    ]

    tile_w, tile_h = tiles[0].size
    canvas_w = 2 * tile_w + gap
    canvas_h = 2 * tile_h + gap

    canvas = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    # Row 1
    canvas.paste(tiles[0], (0, 0))
    canvas.paste(tiles[1], (tile_w + gap, 0))

    # Row 2
    canvas.paste(tiles[2], (0, tile_h + gap))
    canvas.paste(tiles[3], (tile_w + gap, tile_h + gap))

    canvas.save(save_path)


# ============================================================
# 5. Metrics
# ============================================================

def compute_sample_metrics(pred_mask, pred_code, gt_mask, gt_code):
    eps = 1e-6
    result = {}

    if gt_mask is not None:
        pred_mask_f = pred_mask.astype(np.float32)
        gt_mask_f = gt_mask.astype(np.float32)

        inter = (pred_mask_f * gt_mask_f).sum()
        union = ((pred_mask_f + gt_mask_f) > 0).astype(np.float32).sum()
        result["mask_iou"] = float((inter + eps) / (union + eps))

        tp = ((pred_mask == 1) & (gt_mask == 1)).sum()
        fp = ((pred_mask == 1) & (gt_mask == 0)).sum()
        fn = ((pred_mask == 0) & (gt_mask == 1)).sum()

        result["mask_precision"] = float((tp + eps) / (tp + fp + eps))
        result["mask_recall"] = float((tp + eps) / (tp + fn + eps))
        result["mask_f1"] = float((2 * tp + eps) / (2 * tp + fp + fn + eps))

    if gt_mask is not None and gt_code is not None:
        bit_equal = (pred_code == gt_code).astype(np.float32)
        result["code_acc_all_pixels"] = float(bit_equal.mean())

        fg = gt_mask[None, ...].astype(np.float32)
        fg_count = gt_mask.sum()

        if fg_count > 0:
            result["fg_code_acc"] = float((bit_equal * fg).sum() / (fg_count * pred_code.shape[0] + eps))
            all_bits_correct = (bit_equal.mean(axis=0) == 1.0).astype(np.float32)
            result["fg_code_all_correct"] = float((all_bits_correct * gt_mask).sum() / (fg_count + eps))
        else:
            result["fg_code_acc"] = 0.0
            result["fg_code_all_correct"] = 0.0

    return result


# ============================================================
# 6. Inference
# ============================================================

@torch.no_grad()
def predict_sample(model, x, device, threshold=0.5):
    x = x.to(device)
    logits = model(x)
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    pred_mask = preds[:, 0:1].cpu().numpy()[0, 0].astype(np.uint8)
    pred_code = preds[:, 1:].cpu().numpy()[0].astype(np.uint8)

    # Force background code to zero
    pred_code = pred_code * pred_mask[None, :]
    return pred_mask, pred_code


# ============================================================
# 7. Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Predict one sample and save all visualizations.")
    parser.add_argument("--sample_dir", type=str, required=True, help="Path to sample_xxxxxx folder")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pth or last.pth")
    parser.add_argument("--backbone", type=str, default=None, choices=["resnet18", "resnet34"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir", type=str, default=None, help="Default: sample_dir/prediction")
    args = parser.parse_args()

    sample_dir = os.path.abspath(args.sample_dir)
    checkpoint_path = os.path.abspath(args.checkpoint)
    out_dir = os.path.abspath(args.out_dir) if args.out_dir else os.path.join(sample_dir, "prediction")

    os.makedirs(out_dir, exist_ok=True)
    gt_bits_dir = os.path.join(out_dir, "gt_bits")
    pred_bits_dir = os.path.join(out_dir, "pred_bits")
    os.makedirs(gt_bits_dir, exist_ok=True)
    os.makedirs(pred_bits_dir, exist_ok=True)

    device = get_device()
    print(f"[INFO] device = {device}")
    print(f"[INFO] sample_dir = {sample_dir}")
    print(f"[INFO] checkpoint = {checkpoint_path}")
    print(f"[INFO] out_dir = {out_dir}")

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    backbone = args.backbone
    if backbone is None:
        backbone = ckpt.get("args", {}).get("backbone", "resnet18")

    model = ResNetSegmentation(out_channels=11, backbone=backbone, pretrained=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    x, xray_u8 = load_xray_image(sample_dir)
    gt_mask, gt_code_stack = load_optional_gt(sample_dir)

    pred_mask, pred_code_stack = predict_sample(model, x, device, threshold=args.threshold)

    # Save raw outputs
    np.save(os.path.join(out_dir, "pred_mask.npy"), pred_mask.astype(np.uint8))
    np.save(os.path.join(out_dir, "pred_code_stack.npy"), pred_code_stack.astype(np.uint8))
    save_gray_png((pred_mask * 255).astype(np.uint8), os.path.join(out_dir, "pred_mask.png"))

    # Packed code + visualization
    pred_packed_code = pack_code_stack_to_uint16(pred_code_stack)
    np.save(os.path.join(out_dir, "pred_packed_code.npy"), pred_packed_code.astype(np.uint16))

    pred_code_vis = make_code_visualization(pred_packed_code, pred_mask)
    save_gray_png(pred_code_vis, os.path.join(out_dir, "pred_code_vis.png"))

    # Overlay
    pred_overlay = make_mask_overlay(xray_u8, pred_mask)
    save_rgb_png(pred_overlay, os.path.join(out_dir, "pred_overlay_mask.png"))

    gt_code_vis = None
    if gt_mask is not None and gt_code_stack is not None:
        gt_packed_code = pack_code_stack_to_uint16(gt_code_stack)
        gt_code_vis = make_code_visualization(gt_packed_code, gt_mask)

    # Save bit images
    gt_saved = []
    if gt_code_stack is not None and gt_mask is not None:
        gt_saved = save_bit_images(gt_code_stack, gt_mask, gt_bits_dir, prefix="gt")
    pred_saved = save_bit_images(pred_code_stack, pred_mask, pred_bits_dir, prefix="pred")

    # Save 2x10 grid
    if gt_code_stack is not None and gt_mask is not None:
        create_2x10_bit_grid(
            gt_code_stack=gt_code_stack,
            gt_mask=gt_mask,
            pred_code_stack=pred_code_stack,
            pred_mask=pred_mask,
            save_path=os.path.join(out_dir, "bit_grid_2x10.png")
        )

    # Save centered panel
    make_panel_2x2(
        gt_mask=gt_mask,
        gt_code_vis=gt_code_vis,
        pred_mask=pred_mask,
        pred_code_vis=pred_code_vis,
        save_path=os.path.join(out_dir, "pred_vs_gt_panel.png")
    )

    # Metrics
    metrics = compute_sample_metrics(
        pred_mask=pred_mask,
        pred_code=pred_code_stack,
        gt_mask=gt_mask,
        gt_code=gt_code_stack,
    )
    with open(os.path.join(out_dir, "prediction_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("[INFO] Saved outputs:")
    print(f"       {os.path.join(out_dir, 'pred_mask.npy')}")
    print(f"       {os.path.join(out_dir, 'pred_code_stack.npy')}")
    print(f"       {os.path.join(out_dir, 'pred_packed_code.npy')}")
    print(f"       {os.path.join(out_dir, 'pred_mask.png')}")
    print(f"       {os.path.join(out_dir, 'pred_code_vis.png')}")
    print(f"       {os.path.join(out_dir, 'pred_overlay_mask.png')}")
    print(f"       {os.path.join(out_dir, 'pred_vs_gt_panel.png')}")
    if gt_code_stack is not None:
        print(f"       {os.path.join(out_dir, 'bit_grid_2x10.png')}")
    print(f"       {os.path.join(out_dir, 'prediction_metrics.json')}")
    print(f"       {gt_bits_dir}")
    print(f"       {pred_bits_dir}")

    if metrics:
        print("\n[INFO] Per-sample metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.6f}")


if __name__ == "__main__":
    main()



'''

python predict_one_sample.py \
    --sample_dir ./dataset_test/sample_000000 \
    --checkpoint ./checkpoints/best.pth

'''