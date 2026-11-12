"""Mean teacher (C2) on top of the U-Net + EfficientNet-B0 baseline.

Labeled loss: same structure loss as train.py on the mixed TrainDataset.
Unlabeled pool: CVC-ColonDB images only; masks are never read.
Teacher: EMA of the student (no gradients). Consistency: MSE on sigmoid
outputs under two independently flipped views. Consistency weight ramps up
over the first ramp_epochs.

Optional --mc_dropout (C3): T stochastic teacher forwards with dropout on;
per-pixel variance; drop the top uncertainty_drop_frac of pixels from the
consistency loss. Everything else matches plain mean teacher so C3 is alone.

Once ColonDB is used this way, it is no longer an unseen test set for these runs.
"""
import argparse
import copy
import logging
import os
import random
from datetime import datetime

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from train import TrainDataset, structure_loss, eval_dice


class UnlabeledImageDataset(Dataset):
    """Images only. Masks on disk are ignored."""

    def __init__(self, image_root, trainsize=352):
        self.trainsize = trainsize
        self.images = sorted(
            [os.path.join(image_root, f) for f in os.listdir(image_root)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        )
        if len(self.images) == 0:
            raise FileNotFoundError(f'no images in {image_root}')
        self.to_tensor = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.images)

    def _view(self, image):
        if random.random() < 0.5:
            image = TF.hflip(image)
        return self.to_tensor(image)

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        # Two independent weak views of the same frame.
        return self._view(image), self._view(image)


def update_ema(student, teacher, decay):
    with torch.no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data.mul_(decay).add_(s_param.data, alpha=1.0 - decay)
        for s_buf, t_buf in zip(student.buffers(), teacher.buffers()):
            t_buf.data.copy_(s_buf.data)


def consistency_weight(epoch, ramp_epochs, max_weight):
    if ramp_epochs <= 0:
        return max_weight
    return max_weight * min(1.0, float(epoch) / float(ramp_epochs))


@torch.no_grad()
def teacher_mc_predict(teacher, images, t_samples):
    """T stochastic teacher passes. Returns mean and variance of sigmoid maps."""
    preds = []
    for _ in range(t_samples):
        preds.append(torch.sigmoid(teacher(images)))
    stack = torch.stack(preds, dim=0)
    mean = stack.mean(dim=0)
    var = stack.var(dim=0, unbiased=False)
    return mean, var


def consistency_loss(pred_s_logits, teacher, view_t, opt):
    """MSE vs teacher. With --mc_dropout, drop high-variance teacher pixels."""
    pred_s = torch.sigmoid(pred_s_logits)
    if not opt.mc_dropout:
        with torch.no_grad():
            pred_t = torch.sigmoid(teacher(view_t))
        return F.mse_loss(pred_s, pred_t)

    mean_t, var_t = teacher_mc_predict(teacher, view_t, opt.mc_T)
    # Drop the most uncertain fraction of pixels (batch-wide quantile).
    keep_frac = 1.0 - opt.uncertainty_drop_frac
    if keep_frac >= 1.0:
        mask = torch.ones_like(var_t)
    elif keep_frac <= 0.0:
        mask = torch.zeros_like(var_t)
    else:
        thresh = torch.quantile(var_t.flatten(), keep_frac)
        mask = (var_t <= thresh).float()
    denom = mask.sum().clamp_min(1.0)
    return ((pred_s - mean_t).pow(2) * mask).sum() / denom


def train_one_epoch(student, teacher, labeled_loader, unlabeled_loader,
                    optimizer, epoch, opt):
    student.train()
    # Teacher stays in train() so dropout is active for MC Dropout passes.
    teacher.train()
    for p in teacher.parameters():
        p.requires_grad_(False)

    unlabeled_iter = iter(unlabeled_loader)
    size_rates = [0.75, 1, 1.25]
    loss_meter = 0.0
    labeled_meter = 0.0
    cons_meter = 0.0
    n_steps = 0
    w_cons = consistency_weight(epoch, opt.ramp_epochs, opt.consistency_weight)

    for i, (images, gts) in enumerate(labeled_loader, start=1):
        try:
            view_s, view_t = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            view_s, view_t = next(unlabeled_iter)

        for rate in size_rates:
            optimizer.zero_grad()
            images_g = images.cuda(non_blocking=True)
            gts_g = gts.cuda(non_blocking=True)
            view_s_g = view_s.cuda(non_blocking=True)
            view_t_g = view_t.cuda(non_blocking=True)

            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images_g = F.interpolate(images_g, size=(trainsize, trainsize),
                                         mode='bilinear', align_corners=True)
                gts_g = F.interpolate(gts_g, size=(trainsize, trainsize),
                                      mode='bilinear', align_corners=True)
                view_s_g = F.interpolate(view_s_g, size=(trainsize, trainsize),
                                         mode='bilinear', align_corners=True)
                view_t_g = F.interpolate(view_t_g, size=(trainsize, trainsize),
                                         mode='bilinear', align_corners=True)

            pred_l = student(images_g)
            loss_l = structure_loss(pred_l, gts_g)

            pred_s = student(view_s_g)
            loss_c = consistency_loss(pred_s, teacher, view_t_g, opt)

            loss = loss_l + w_cons * loss_c
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), opt.clip)
            optimizer.step()
            update_ema(student, teacher, opt.ema_decay)

            if rate == 1:
                loss_meter += loss.item()
                labeled_meter += loss_l.item()
                cons_meter += loss_c.item()
                n_steps += 1

        if i % 20 == 0 or i == len(labeled_loader):
            n = max(n_steps, 1)
            print(
                '{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], '
                'loss: {:.4f} (lab: {:.4f} cons: {:.4f} w: {:.3f})'.format(
                    datetime.now(), epoch, opt.epoch, i, len(labeled_loader),
                    loss_meter / n, labeled_meter / n, cons_meter / n, w_cons)
            )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--trainsize', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_path', type=str, default='../datasets/TrainDataset')
    parser.add_argument('--unlabeled_path', type=str,
                        default='../datasets/TestDataset/CVC-ColonDB/images')
    parser.add_argument('--test_path', type=str, default='../datasets/TestDataset')
    parser.add_argument('--save_dir', type=str, default='./runs/mean_teacher')
    parser.add_argument('--ema_decay', type=float, default=0.99)
    parser.add_argument('--consistency_weight', type=float, default=0.1)
    parser.add_argument('--ramp_epochs', type=int, default=5,
                        help='epochs to ramp consistency weight from 0 to max')
    parser.add_argument('--mc_dropout', action='store_true',
                        help='C3: filter consistency by teacher MC Dropout variance')
    parser.add_argument('--mc_T', type=int, default=10,
                        help='number of stochastic teacher passes')
    parser.add_argument('--uncertainty_drop_frac', type=float, default=0.5,
                        help='fraction of highest-variance teacher pixels to drop')
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed:', opt.seed)
    print('ema_decay:', opt.ema_decay,
          'consistency_weight:', opt.consistency_weight,
          'ramp_epochs:', opt.ramp_epochs)
    print('mc_dropout:', opt.mc_dropout,
          'mc_T:', opt.mc_T,
          'uncertainty_drop_frac:', opt.uncertainty_drop_frac)
    print('unlabeled_path:', opt.unlabeled_path)
    print('NOTE: CVC-ColonDB is used unlabeled; treat it as seen for these runs.')

    run_dir = os.path.join(opt.save_dir, f'seed{opt.seed}')
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(run_dir, f'train_seed{opt.seed}.log'),
        format='[%(asctime)s-%(levelname)s:%(message)s]',
        level=logging.INFO, filemode='w',
        datefmt='%Y-%m-%d %I:%M:%S %p',
    )

    student = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
    ).cuda()
    teacher = copy.deepcopy(student).cuda()
    for p in teacher.parameters():
        p.requires_grad_(False)

    n_params = sum(p.numel() for p in student.parameters()) / 1e6
    print(f'params (M): {n_params:.3f}')
    n_drop = sum(1 for m in student.modules() if isinstance(m, torch.nn.Dropout))
    print(f'dropout_modules: {n_drop}')

    optimizer = torch.optim.AdamW(student.parameters(), lr=opt.lr, weight_decay=1e-4)

    labeled_set = TrainDataset(
        os.path.join(opt.train_path, 'images'),
        os.path.join(opt.train_path, 'masks'),
        opt.trainsize,
    )
    unlabeled_set = UnlabeledImageDataset(opt.unlabeled_path, opt.trainsize)
    print(f'n_labeled={len(labeled_set)} n_unlabeled={len(unlabeled_set)}')

    labeled_loader = DataLoader(
        labeled_set, batch_size=opt.batchsize, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    unlabeled_loader = DataLoader(
        unlabeled_set, batch_size=opt.batchsize, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )

    # ColonDB is seen (unlabeled) for these runs; still log its Dice for the report.
    datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    best = 0.0
    title = 'Start Mean-Teacher + MC Dropout' if opt.mc_dropout else 'Start Mean-Teacher Training'
    print('#' * 20, title, '#' * 20)

    for epoch in range(1, opt.epoch + 1):
        train_one_epoch(
            student, teacher, labeled_loader, unlabeled_loader, optimizer, epoch, opt,
        )
        scores = {}
        for name in datasets:
            dice = eval_dice(student, opt.test_path, name, opt.trainsize)
            scores[name] = dice
            print(f'{name}: {dice:.6f}')
            logging.info('epoch: %s dataset: %s dice: %s', epoch, name, dice)
        meandice = (scores['Kvasir'] + scores['CVC-ClinicDB']) / 2.0
        print(f'in-domain mean (Kvasir+ClinicDB): {meandice:.6f}')
        torch.save(student.state_dict(), os.path.join(run_dir, 'UNet_last.pth'))
        torch.save(teacher.state_dict(), os.path.join(run_dir, 'UNet_teacher_last.pth'))
        if meandice > best:
            best = meandice
            torch.save(student.state_dict(), os.path.join(run_dir, 'UNet_best.pth'))
            torch.save(teacher.state_dict(), os.path.join(run_dir, 'UNet_teacher_best.pth'))
            print('best', best)
            logging.info('best: %s', best)

    print('done. best in-domain mean:', best)

