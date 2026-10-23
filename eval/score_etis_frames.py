"""Per-frame ETIS Dice for the checkpoints used in the qualitative figure.

Seeds are the ones whose small-polyp ETIS Dice is closest to that model's
three-seed mean: Polyp-PVT seed 0, PraNet seed 1, U-Net seed 0.
Oversampling is the only completed run, U-Net seed 0.
"""
import csv
import os
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from size_split_metrics import dice, load_model, predict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ETIS = os.path.join(ROOT, "datasets", "TestDataset", "ETIS-LaribPolypDB")
OUT = os.path.join(ROOT, "eval", "etis_frame_dice.csv")
THRESHOLD = 20957

JOBS = [
    ("polyp_pvt", 0, "Polyp-PVT/runs/seed0/PolypPVT.pth"),
    ("pranet", 1, "PraNet/runs/seed1/PraNet-19.pth"),
    ("unet", 0, "UNet/runs/seed0/UNet_best.pth"),
    ("unet_oversample", 0, "UNet/runs/oversample_c1/seed0/UNet_best.pth"),
]


def frames():
    image_root = os.path.join(ETIS, "images")
    mask_root = os.path.join(ETIS, "masks")
    names = sorted(
        f for f in os.listdir(image_root)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    )
    transform = transforms.Compose([
        transforms.Resize((352, 352)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    for name in names:
        image = transform(Image.open(os.path.join(image_root, name)).convert("RGB")).unsqueeze(0)
        gt = np.asarray(Image.open(os.path.join(mask_root, name)).convert("L"), dtype=np.float32)
        gt /= (gt.max() + 1e-8)
        yield name, image, gt


def main():
    cached = list(frames())
    rows = []
    for model_name, seed, pth in JOBS:
        kind = "unet" if model_name.startswith("unet") else model_name
        print("loading", model_name, pth, flush=True)
        model = load_model(kind, os.path.join(ROOT, pth))
        dices = []
        for name, image, gt in cached:
            pred = predict(model, kind, image, gt.shape)
            d = dice(pred, gt)
            area = int((gt > 0.5).sum())
            dices.append(d)
            rows.append({
                "model": model_name,
                "seed": seed,
                "frame": name,
                "area": area,
                "group": "small" if area <= THRESHOLD else "large",
                "dice": f"{d:.6f}",
            })
        print(f"  ETIS mean dice {np.mean(dices):.4f}", flush=True)
        del model
        torch.cuda.empty_cache()
    with open(OUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "seed", "frame", "area", "group", "dice"])
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
