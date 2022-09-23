
import torch
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder


def Loader(args):

    # Data transform
    trn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    tst_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])

    # Load Dataset
    trnset = ImageFolder(root=f'{args.root}/cifar60/train', transform=trn_transform)
    trnloader = DataLoader(trnset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    tstset = ImageFolder(root=f'{args.root}/cifar60/eval',transform=tst_transform)
    tstloader = DataLoader(tstset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    return trnloader, tstloader