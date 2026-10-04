"""Compute mean Dice for a PraNet checkpoint on the standard test sets."""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from lib.PraNet_Res2Net import PraNet
from utils.dataloader import test_dataset


def mean_dice(model, data_path, testsize=352):
    image_root = os.path.join(data_path, 'images')
    gt_root = os.path.join(data_path, 'masks')
    num = len(os.listdir(gt_root))
    loader = test_dataset(image_root + '/', gt_root + '/', testsize)
    total = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(num):
            image, gt, _ = loader.load_data()
            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            res5, res4, res3, res2 = model(image)
            res = F.interpolate(res2, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            smooth = 1.0
            intersection = (res * gt).sum()
            dice = (2 * intersection + smooth) / (res.sum() + gt.sum() + smooth)
            total += float(dice)
    return total / num


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pth_path', type=str, required=True)
    parser.add_argument('--test_path', type=str, default='../datasets/TestDataset')
    parser.add_argument('--testsize', type=int, default=352)
    opt = parser.parse_args()

    model = PraNet().cuda()
    model.load_state_dict(torch.load(opt.pth_path, map_location='cuda'))
    print('loaded', opt.pth_path)

    datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    scores = {}
    for name in datasets:
        path = os.path.join(opt.test_path, name)
        dice = mean_dice(model, path, opt.testsize)
        scores[name] = dice
        print(f'{name}: {dice:.6f}')

    in_domain = (scores['Kvasir'] + scores['CVC-ClinicDB']) / 2.0
    print(f'in-domain mean (Kvasir+ClinicDB): {in_domain:.6f}')
