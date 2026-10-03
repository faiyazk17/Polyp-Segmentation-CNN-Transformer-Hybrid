"""Lightweight U-Net + EfficientNet-B0 baseline for polyp segmentation."""
import argparse
import logging
import os
import random
from datetime import datetime

import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
from PIL import Image
import torchvision.transforms as transforms


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, image_root, gt_root, trainsize=352):
        self.trainsize = trainsize
        self.images = sorted(
            [os.path.join(image_root, f) for f in os.listdir(image_root)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        )
        self.gts = sorted(
            [os.path.join(gt_root, f) for f in os.listdir(gt_root)
             if f.lower().endswith(('.png', '.jpg'))]
        )
        assert len(self.images) == len(self.gts), 'image/mask count mismatch'
        self.img_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.gt_transform = transforms.Compose([
            transforms.Resize((trainsize, trainsize)),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert('RGB')
        gt = Image.open(self.gts[index]).convert('L')
        return self.img_transform(image), self.gt_transform(gt)

    def __len__(self):
        return len(self.images)


class TestDataset:
    def __init__(self, image_root, gt_root, testsize=352):
        self.testsize = testsize
        self.images = sorted(
            [os.path.join(image_root, f) for f in os.listdir(image_root)
             if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        )
        self.gts = sorted(
            [os.path.join(gt_root, f) for f in os.listdir(gt_root)
             if f.lower().endswith(('.png', '.jpg'))]
        )
        self.size = len(self.images)
        self.index = 0
        self.transform = transforms.Compose([
            transforms.Resize((testsize, testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def load_data(self):
        image = self.transform(Image.open(self.images[self.index]).convert('RGB')).unsqueeze(0)
        gt = Image.open(self.gts[self.index]).convert('L')
        name = os.path.basename(self.images[self.index])
        self.index += 1
        return image, gt, name


def structure_loss(pred, mask):
    weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
    wbce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))
    pred = torch.sigmoid(pred)
    inter = ((pred * mask) * weit).sum(dim=(2, 3))
    union = ((pred + mask) * weit).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


@torch.no_grad()
def eval_dice(model, test_root, dataset, testsize=352):
    data_path = os.path.join(test_root, dataset)
    loader = TestDataset(
        os.path.join(data_path, 'images'),
        os.path.join(data_path, 'masks'),
        testsize,
    )
    model.eval()
    total = 0.0
    for _ in range(loader.size):
        image, gt, _ = loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        pred = model(image)
        pred = F.interpolate(pred, size=gt.shape, mode='bilinear', align_corners=False)
        pred = pred.sigmoid().data.cpu().numpy().squeeze()
        pred = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
        smooth = 1.0
        dice = (2 * (pred * gt).sum() + smooth) / (pred.sum() + gt.sum() + smooth)
        total += float(dice)
    return total / loader.size


def train_one_epoch(model, loader, optimizer, epoch, opt):
    model.train()
    size_rates = [0.75, 1, 1.25]
    loss_meter = 0.0
    n_steps = 0
    for i, (images, gts) in enumerate(loader, start=1):
        for rate in size_rates:
            optimizer.zero_grad()
            images_g = Variable(images).cuda()
            gts_g = Variable(gts).cuda()
            trainsize = int(round(opt.trainsize * rate / 32) * 32)
            if rate != 1:
                images_g = F.interpolate(images_g, size=(trainsize, trainsize),
                                         mode='bilinear', align_corners=True)
                gts_g = F.interpolate(gts_g, size=(trainsize, trainsize),
                                      mode='bilinear', align_corners=True)
            pred = model(images_g)
            loss = structure_loss(pred, gts_g)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
            optimizer.step()
            if rate == 1:
                loss_meter += loss.item()
                n_steps += 1
        if i % 20 == 0 or i == len(loader):
            avg = loss_meter / max(n_steps, 1)
            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], loss: {:.4f}'.format(
                datetime.now(), epoch, opt.epoch, i, len(loader), avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batchsize', type=int, default=16)
    parser.add_argument('--trainsize', type=int, default=352)
    parser.add_argument('--clip', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train_path', type=str, default='../datasets/TrainDataset')
    parser.add_argument('--test_path', type=str, default='../datasets/TestDataset')
    parser.add_argument('--save_dir', type=str, default='./runs')
    opt = parser.parse_args()

    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print('seed:', opt.seed)

    run_dir = os.path.join(opt.save_dir, f'seed{opt.seed}')
    os.makedirs(run_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(run_dir, f'train_seed{opt.seed}.log'),
        format='[%(asctime)s-%(levelname)s:%(message)s]',
        level=logging.INFO, filemode='w',
        datefmt='%Y-%m-%d %I:%M:%S %p',
    )

    model = smp.Unet(
        encoder_name='efficientnet-b0',
        encoder_weights='imagenet',
        in_channels=3,
        classes=1,
    ).cuda()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'params (M): {n_params:.3f}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    train_set = TrainDataset(
        os.path.join(opt.train_path, 'images'),
        os.path.join(opt.train_path, 'masks'),
        opt.trainsize,
    )
    train_loader = DataLoader(
        train_set, batch_size=opt.batchsize, shuffle=True,
        num_workers=4, pin_memory=True,
    )

    datasets = ['CVC-300', 'CVC-ClinicDB', 'Kvasir', 'CVC-ColonDB', 'ETIS-LaribPolypDB']
    best = 0.0
    print('#' * 20, 'Start Training', '#' * 20)

    for epoch in range(1, opt.epoch + 1):
        train_one_epoch(model, train_loader, optimizer, epoch, opt)
        scores = {}
        for name in datasets:
            dice = eval_dice(model, opt.test_path, name, opt.trainsize)
            scores[name] = dice
            print(f'{name}: {dice:.6f}')
            logging.info('epoch: %s dataset: %s dice: %s', epoch, name, dice)
        meandice = (scores['Kvasir'] + scores['CVC-ClinicDB']) / 2.0
        print(f'in-domain mean (Kvasir+ClinicDB): {meandice:.6f}')
        torch.save(model.state_dict(), os.path.join(run_dir, 'UNet_last.pth'))
        if meandice > best:
            best = meandice
            torch.save(model.state_dict(), os.path.join(run_dir, 'UNet_best.pth'))
            print('best', best)
            logging.info('best: %s', best)

    print('done. best in-domain mean:', best)
