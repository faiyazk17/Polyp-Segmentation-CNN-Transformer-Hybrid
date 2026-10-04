"""
Week 3–4 metrics: mean Dice, mean IoU, weighted F_beta for all baselines.

Usage (from repo root):
  python eval/full_metrics.py --model polyp_pvt --pth Polyp-PVT/runs/seed0/PolypPVT.pth
  python eval/full_metrics.py --run_all
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter
import torchvision.transforms as transforms

DATASETS = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']


def weighted_f_beta(pred, gt, beta=1.0):
    """Weighted F-measure (Margolin et al., CVPR'14) — same family as PraNet MATLAB WFb."""
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    if pred.max() > 1.0 or pred.min() < 0.0:
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
    gt_mask = gt > 0.5
    not_gt_mask = ~gt_mask
    gt_f = gt_mask.astype(np.float64)

    if not np.any(gt_mask):
        return 0.0 if np.any(pred > 0.5) else 1.0

    E = np.abs(pred - gt_f)
    dist, idx = distance_transform_edt(not_gt_mask, return_indices=True)
    Et = E.copy()
    Et[not_gt_mask] = E[idx[0, not_gt_mask], idx[1, not_gt_mask]]
    EA = gaussian_filter(Et, sigma=5.0, truncate=3 / 5.0, mode='constant', cval=0.0)
    min_E_EA = np.where(gt_mask, np.minimum(E, EA), E)

    B = np.ones_like(gt_f)
    B[not_gt_mask] = 2.0 - np.exp(np.log(0.5) / 5.0 * dist[not_gt_mask])
    Ew = min_E_EA * B

    eps = np.finfo(np.float64).eps
    TPw = np.sum(gt_f) - np.sum(Ew[gt_mask])
    FPw = np.sum(Ew[not_gt_mask])
    R = 1.0 - np.mean(Ew[gt_mask])
    P = TPw / (eps + TPw + FPw)
    return float((1 + beta ** 2) * (R * P) / (eps + R + (beta ** 2) * P))


def dice_iou(pred, gt, smooth=1.0):
    pred = pred.reshape(-1)
    gt = gt.reshape(-1)
    inter = (pred * gt).sum()
    dice = (2 * inter + smooth) / (pred.sum() + gt.sum() + smooth)
    union = pred.sum() + gt.sum() - inter
    iou = (inter + smooth) / (union + smooth)
    return float(dice), float(iou)


class TestLoader:
    def __init__(self, image_root, gt_root, testsize=352):
        self.testsize = testsize
        self.images = sorted([
            os.path.join(image_root, f) for f in os.listdir(image_root)
            if f.lower().endswith(('.jpg', '.png', '.jpeg'))
        ])
        self.gts = sorted([
            os.path.join(gt_root, f) for f in os.listdir(gt_root)
            if f.lower().endswith(('.png', '.jpg'))
        ])
        self.size = len(self.images)
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
            yield image, gt


def _clear_lib_imports():
    for key in list(sys.modules):
        if key == 'lib' or key.startswith('lib.'):
            del sys.modules[key]


def load_model(model_name, pth_path):
    repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    pth_path = pth_path if os.path.isabs(pth_path) else os.path.join(repo, pth_path)
    _clear_lib_imports()
    # Keep only one model package root on path at a time
    sys.path = [p for p in sys.path if not p.endswith('Polyp-PVT') and not p.endswith('PraNet')]

    # Polyp-PVT / PraNet constructors load backbones via paths relative to CWD
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


def eval_checkpoint(model_name, pth_path, test_root, testsize=352):
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
        dices, ious, wfbs = [], [], []
        for image, gt in loader:
            pred = predict(model, model_name, image, gt.shape)
            d, iou = dice_iou(pred, gt)
            wfb = weighted_f_beta(pred, gt, beta=1.0)
            dices.append(d)
            ious.append(iou)
            wfbs.append(wfb)
        row = {
            'dataset': name,
            'mDice': float(np.mean(dices)),
            'mIoU': float(np.mean(ious)),
            'wFb': float(np.mean(wfbs)),
        }
        rows.append(row)
        print(f"{name}: mDice={row['mDice']:.4f}  mIoU={row['mIoU']:.4f}  wFb={row['wFb']:.4f}")
    clinic = next(r for r in rows if r['dataset'] == 'CVC-ClinicDB')
    kvasir = next(r for r in rows if r['dataset'] == 'Kvasir')
    print(f"in-domain mean mDice: {(clinic['mDice'] + kvasir['mDice']) / 2:.4f}")
    return rows


def run_all(test_root, out_csv):
    jobs = []
    for s in (0, 1, 2):
        jobs.append(('polyp_pvt', s, f'Polyp-PVT/runs/seed{s}/PolypPVT.pth'))
        jobs.append(('pranet', s, f'PraNet/runs/seed{s}/PraNet-19.pth'))
        jobs.append(('unet', s, f'UNet/runs/seed{s}/UNet_best.pth'))

    all_rows = []
    for model_name, seed, pth in jobs:
        if not os.path.isfile(pth):
            print(f'SKIP missing {pth}')
            continue
        print(f'\n===== {model_name} seed {seed} =====')
        print('loaded', pth)
        rows = eval_checkpoint(model_name, pth, test_root)
        for r in rows:
            all_rows.append({
                'model': model_name,
                'seed': seed,
                **r,
            })

    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['model', 'seed', 'dataset', 'mDice', 'mIoU', 'wFb'])
        writer.writeheader()
        writer.writerows(all_rows)
    print('\nWrote', out_csv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['polyp_pvt', 'pranet', 'unet'])
    parser.add_argument('--pth', type=str)
    parser.add_argument('--test_path', type=str, default='datasets/TestDataset')
    parser.add_argument('--run_all', action='store_true')
    parser.add_argument('--out_csv', type=str, default='eval/week3_4_metrics.csv')
    args = parser.parse_args()

    if args.run_all:
        run_all(args.test_path, args.out_csv)
    else:
        if not args.model or not args.pth:
            parser.error('--model and --pth required unless --run_all')
        eval_checkpoint(args.model, args.pth, args.test_path)
