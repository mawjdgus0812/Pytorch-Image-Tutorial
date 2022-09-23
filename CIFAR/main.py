import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision

from tqdm import tqdm

# dataset.py
from dataset import Loader

# resnet.py
from resnet import resnet18

from train import train
from eval import evaluate

import utils

# parser
parser = argparse.ArgumentParser()

parser.add_argument('--root', default='/data/cifar100-based', type=str)
parser.add_argument('--save_dir', default='./saved/', type=str)
parser.add_argument('--trial', default='01', type=str)

parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)



args = parser.parse_args()

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# main
def main():

    torch.backends.cudnn.deterministic = True
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    save_dir = args.save_dir + f'{args.arch}-{args.trial}'
    os.makedirs(save_dir,exist_ok=True)
    utils.save_exp(args, save_dir)

    # DataLoader
    trnloader, tstloader = Loader(args)

    # model 
    model = resnet18(num_classes = 60).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9, nesterov=False)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=[120, 160], gamma=0.1)

    results = []

    for epoch in tqdm(range(args.epoch)):
        train(args, trnloader, model, optimizer, criterion, epoch)
        result = evaluate(tstloader, model, criterion, epoch)
        results.append(result)
        scheduler.step()
        # if epoch % 50 == 49:
        print(np.max(results))

if __name__=='__main__':
    main()
