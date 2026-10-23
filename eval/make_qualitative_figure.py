"""Representative ETIS figure: one small polyp and one large polyp.

The cut is the frozen training-set median, 20,957 foreground pixels.
Chosen frames: 112.png (small) and 163.png (large).
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ETIS = os.path.join(ROOT, "datasets", "TestDataset", "ETIS-LaribPolypDB")
OUT = os.path.join(ROOT, "docs", "figures", "etis_size_examples.png")
THRESHOLD = 20957
CASES = [("112.png", "Small"), ("163.png", "Large")]


def load_case(name):
    image = np.array(Image.open(os.path.join(ETIS, "images", name)).convert("RGB"))
    mask = np.array(Image.open(os.path.join(ETIS, "masks", name)).convert("L")) > 0
    return image, mask


def outline(image, mask, width=4):
    edge = ndimage.binary_dilation(mask, iterations=width) & ~mask
    out = image.copy()
    out[edge] = (200, 30, 40)
    return out


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for row, (name, label) in enumerate(CASES):
        image, mask = load_case(name)
        area = int(mask.sum())
        axes[row, 0].imshow(image)
        axes[row, 1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        axes[row, 2].imshow(outline(image, mask))
        axes[row, 0].set_ylabel(
            f"{label}\n{area:,} px",
            fontsize=9,
            rotation=90,
            labelpad=8,
        )
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])
        group = "small" if area <= THRESHOLD else "large"
        print(f"{name}: area={area} -> {group}")
    for ax, title in zip(axes[0], ("Frame", "Expert mask", "Outline")):
        ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
