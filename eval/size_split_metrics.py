"""
Week 5: overall / small / large mean Dice using frozen area threshold.

Usage (from repo root):
  python eval/size_split_metrics.py --model unet --pth UNet/runs/seed0/UNet_best.pth
  python eval/size_split_metrics.py --run_all
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

DATASETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


def load_threshold(bins_path):
    with open(bins_path, 'r') as f:
        cfg = json.load(f)
    return int(cfg['median_threshold']), cfg


def polyp_area(gt):
    return int((gt > 0.5).sum())


def dice(pred, gt, smooth=1.0):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    inter = (pred * gt).sum()
    return float((2 * inter + smooth) / (pred.sum() + gt.sum() + smooth))


class TestLoader:
    def __init__(self, image_root, gt_root, testsize=352):
        self.images = sorted([
            os.path.join(image_root, f) for f in os.listdir(image_root)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.gts = sorted([
            os.path.join(gt_root, f) for f in os.listdir(gt_root)
            if f.lower().endswith(('.png', '.jpg'))
        ])
        self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __iter__(self):
        for img_path, gt_path in zip(self.images, self.gts):
            image = self.transform(Image.open(img_path).convert('RGB')).unsqueeze(0)
            gt = np.asarray(Image.open(gt_path).convert('L'), dtype=np.float32)
            gt /= (gt.max() + 1e-8)
            yield image, gt, os.path.basename(gt_path)


def _clear_lib_imports():
    for key in list(sys.modules):
        if key == 'lib' or key.startswith('lib.'):
            del sys.modules[key]


def load_model(model_name, pth_path):
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pth_path = pth_path if os.path.isabs(pth_path) else os.path.join(repo, pth_path)
    _clear_lib_imports()
    sys.path = [p for p in sys.path if not p.endswith('Polyp-PVT') and not p.endswith('PraNet')]
    prev_cwd = os.getcwd()
    try:
        if model_name == 'polyp_pvt':
            model_dir = os.path.join(repo, 'Polyp-PVT')
            sys.path.insert(0, model_dir)
            os.chdir(model_dir)
            from lib.pvt import PolypPVT
            model = PolypPVT().cuda()
            model.load_state_dict(torch.load(pth_path, map_location='cuda'))
        elif model_name == 'pranet':
            model_dir = os.path.join(repo, 'PraNet')
            sys.path.insert(0, model_dir)
            os.chdir(model_dir)
            from lib.PraNet_Res2Net import PraNet
            model = PraNet().cuda()
            model.load_state_dict(torch.load(pth_path, map_location='cuda'))
        elif model_name == 'unet':
            import segmentation_models_pytorch as smp
            model = smp.Unet(
                encoder_name='efficientnet-b0',
                encoder_weights=None,
                in_channels=3,
                classes=1,
            ).cuda()
            model.load_state_dict(torch.load(pth_path, map_location='cuda'))
        else:
            raise ValueError(model_name)
    finally:
        os.chdir(prev_cwd)
    model.eval()
    return model


@torch.no_grad()
def predict(model, model_name, image, gt_shape):
    image = image.cuda()
    if model_name == 'polyp_pvt':
        p1, p2 = model(image)
        pred = F.interpolate(p1 + p2, size=gt_shape, mode='bilinear', align_corners=False)
    elif model_name == 'pranet':
        _, _, _, res2 = model(image)
        pred = F.interpolate(res2, size=gt_shape, mode='bilinear', align_corners=False)
    else:
        pred = F.interpolate(model(image), size=gt_shape, mode='bilinear', align_corners=False)
    pred = pred.sigmoid().cpu().numpy().squeeze()
    pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    return pred


def eval_checkpoint(model_name, pth_path, test_root, threshold, testsize=352):
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if not os.path.isabs(test_root):
        test_root = os.path.join(repo, test_root)
    model = load_model(model_name, pth_path)
    rows = []
    for name in DATASETS:
        loader = TestLoader(
            os.path.join(test_root, name, 'images'),
            os.path.join(test_root, name, 'masks'),
            testsize,
        )
        all_d, small_d, large_d = [], [], []
        for image, gt, _ in loader:
            area = polyp_area(gt)
            pred = predict(model, model_name, image, gt.shape)
            d = dice(pred, gt)
            all_d.append(d)
            if area <= threshold:
                small_d.append(d)
            else:
                large_d.append(d)
        row = {
            'dataset': name,
            'n_all': len(all_d),
            'n_small': len(small_d),
            'n_large': len(large_d),
            'dice_all': float(np.mean(all_d)) if all_d else float('nan'),
            'dice_small': float(np.mean(small_d)) if small_d else float('nan'),
            'dice_large': float(np.mean(large_d)) if large_d else float('nan'),
        }
        rows.append(row)
        print(
            f"{name}: all={row['dice_all']:.4f} (n={row['n_all']})  "
            f"small={row['dice_small']:.4f} (n={row['n_small']})  "
            f"large={row['dice_large']:.4f} (n={row['n_large']})"
        )
    return rows


def run_all(test_root, threshold, out_csv):
    jobs = []
    for s in (0, 1, 2):
        jobs.append(('polyp_pvt', s, f'Polyp-PVT/runs/seed{s}/PolypPVT.pth'))
        jobs.append(('pranet', s, f'PraNet/runs/seed{s}/PraNet-19.pth'))
        jobs.append(('unet', s, f'UNet/runs/seed{s}/UNet_best.pth'))

    all_rows = []
    for model_name, seed, pth in jobs:
        if not os.path.isfile(pth) and not os.path.isfile(
            os.path.join(os.path.dirname(__file__), '..', pth)
        ):
            # resolve relative to repo
            repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            full = os.path.join(repo, pth)
            if not os.path.isfile(full):
                print(f'SKIP missing {pth}')
                continue
        print(f'\n===== {model_name} seed {seed} =====')
        print('loaded', pth)
        rows = eval_checkpoint(model_name, pth, test_root, threshold)
        for r in rows:
            all_rows.append({'model': model_name, 'seed': seed, 'threshold': threshold, **r})

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                'model', 'seed', 'threshold', 'dataset',
                'n_all', 'n_small', 'n_large',
                'dice_all', 'dice_small', 'dice_large',
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)
    print('\nWrote', out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['polyp_pvt', 'pranet', 'unet'])
    parser.add_argument('--pth', type=str)
    parser.add_argument('--test_path', type=str, default='datasets/TestDataset')
    parser.add_argument('--bins', type=str, default='eval/size_bins.json')
    parser.add_argument('--run_all', action='store_true')
    parser.add_argument('--out_csv', type=str, default='eval/week5_size_split.csv')
    args = parser.parse_args()

    bins_path = args.bins
    if not os.path.isabs(bins_path):
        repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        bins_path = os.path.join(repo, bins_path)
    threshold, cfg = load_threshold(bins_path)
    print('threshold:', threshold, '|', cfg['small'], '|', cfg['large'])

    if args.run_all:
        run_all(args.test_path, threshold, args.out_csv)
    else:
        if not args.model or not args.pth:
            parser.error('--model and --pth required unless --run_all')
        eval_checkpoint(args.model, args.pth, args.test_path, threshold)
