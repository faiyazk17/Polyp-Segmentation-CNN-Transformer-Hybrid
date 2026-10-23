"""ETIS qualitative figure: predictions for one small frame and one large frame.

Frames are the ETIS images whose Polyp-PVT, PraNet, and U-Net Dice are jointly
closest to those checkpoints' own small-group and large-group means, among
frames that are large enough to see. Seeds are the ones closest to each
model's three-seed small-polyp mean.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from scipy import ndimage

sys.path.insert(0, os.path.dirname(__file__))
from size_split_metrics import dice, load_model, predict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ETIS = os.path.join(ROOT, "datasets", "TestDataset", "ETIS-LaribPolypDB")
OUT = os.path.join(ROOT, "docs", "figures", "etis_predictions.png")

CASES = [
    ("101.png", "Small"),
    ("35.png", "Large"),
]
MODELS = [
    ("Polyp-PVT\nseed 0", "polyp_pvt", "Polyp-PVT/runs/seed0/PolypPVT.pth"),
    ("PraNet\nseed 1", "pranet", "PraNet/runs/seed1/PraNet-19.pth"),
    ("U-Net\nseed 0", "unet", "UNet/runs/seed0/UNet_best.pth"),
    ("U-Net\noversampled", "unet", "UNet/runs/oversample_c1/seed0/UNet_best.pth"),
]


def load_frame(name):
    image = np.array(Image.open(os.path.join(ETIS, "images", name)).convert("RGB"))
    gt = np.asarray(Image.open(os.path.join(ETIS, "masks", name)).convert("L"), dtype=np.float32)
    gt /= (gt.max() + 1e-8)
    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(Image.fromarray(image)).unsqueeze(0)
    return image, gt, tensor


def contour(mask, width=3):
    mask = mask.astype(bool)
    return ndimage.binary_dilation(mask, iterations=width) & ~mask


def draw(ax, image, gt, pred=None):
    canvas = image.copy()
    gt_edge = contour(gt > 0.5)
    canvas[gt_edge] = (40, 170, 60)
    if pred is not None:
        pred_edge = contour(pred > 0.5, width=3)
        # prediction contour wins where the two outlines meet, so both stay visible
        canvas[pred_edge] = (200, 30, 40)
    ax.imshow(canvas)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    loaded = []
    for title, kind, pth in MODELS:
        print("loading", title.replace("\n", " "), flush=True)
        model = load_model(kind, os.path.join(ROOT, pth))
        loaded.append((title, kind, model))

    fig, axes = plt.subplots(2, 5, figsize=(10.2, 4.6))
    for row, (name, label) in enumerate(CASES):
        image, gt, tensor = load_frame(name)
        area = int((gt > 0.5).sum())
        draw(axes[row, 0], image, gt)
        axes[row, 0].set_ylabel(f"{label}\n{name.replace('.png', '')}\n{area:,} px", fontsize=8)
        if row == 0:
            axes[row, 0].set_title("Frame\nexpert outline", fontsize=9)
        for col, (title, kind, model) in enumerate(loaded, start=1):
            pred = predict(model, kind, tensor, gt.shape)
            d = dice(pred, gt)
            draw(axes[row, col], image, gt, pred)
            if row == 0:
                axes[row, col].set_title(title, fontsize=9)
            axes[row, col].set_xlabel(f"Dice {d:.3f}", fontsize=8)
            print(f"{name} {title.replace(chr(10), ' ')} dice={d:.3f} area={area}")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)
    for _, _, model in loaded:
        del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
