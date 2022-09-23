
import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder


def Loader(args):

    mean = [0.4914, 0.4822, 0.4465]
    stdv = [0.247, 0.243, 0.261]

    # Data transform
    trn_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=stdv),
    ])


    tst_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=stdv),
    ])

    # Load Dataset
    trnset = ImageFolder(root=f'{args.root}/cifar60/train', transform=trn_transform)
    trnloader = DataLoader(trnset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    tstset = ImageFolder(root=f'{args.root}/cifar60/eval',transform=tst_transform)
    tstloader = DataLoader(tstset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return trnloader, tstloader
