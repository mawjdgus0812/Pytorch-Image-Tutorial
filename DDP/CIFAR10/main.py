import os
import glob
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

import time

import torchvision
import torchvision.transforms as transforms

import tqdm
import argparse

from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--root', default='/data/mawjdgus', type=str)
parser.add_argument('--save_path', default='./saved/', type=str)
# parser.add_argument('--trial', default='01', type=str)

# parser.add_argument('--arch', default='resnet18', type=str)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epoch', default=200, type=int)
parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--world-size', default=4, type=int)

args = parser.parse_args()

def main(rank, args):
    # Initialize Each Process
    init_process(rank, args.world_size)
    
    # Dataset & DataLoader
    data_path = glob.glob('/data/mawjdgus')[0]
    
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010))
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                                 train=True,
                                                 transform=train_transform,
                                                 download=False)
    
    test_dataset = torchvision.datasets.CIFAR10(root=data_path,
                                                 train=False,
                                                 transform=val_transform,
                                                 download=False)
    
    args.batch_size = int(args.batch_size / args.world_size)
    
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    rank=rank,
                                                                    num_replicas=args.world_size,
                                                                    shuffle=True)
    val_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                  rank=rank,
                                                                  num_replicas=args.world_size,
                                                                  shuffle=False)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               sampler=train_sampler)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=False,
                                               num_workers=4,
                                               sampler=val_sampler)
    
    model = torchvision.models.__dict__['resnet18'](num_classes=10, pretrained=False)
    torch.cuda.set_device(rank)
    model = model.cuda(rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    
    criterion = nn.CrossEntropyLoss().cuda(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1,
                                momentum=0.9, weight_decay=1e-4)
    
    train_logger = Logger(os.path.join(args.save_path, 'train.log'))
    test_logger = Logger(os.path.join(args.save_path, 'test.log'))    
    
    for epoch in range(args.epoch):
        train_sampler.set_epoch(epoch)
        train(model, criterion, optimizer, train_loader,train_logger, epoch, args)
        test(model, criterion, test_loader, test_logger, epoch, args)
        
def train(model, criterion, optimizer, train_loader,train_logger, epoch, args):
    model.train()
    
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    iter_time = AverageMeter()
    data_time = AverageMeter()
    
    end = time.time()
    for i, (data, target) in enumerate(train_loader):
        data, target = data.cuda(dist.get_rank()), target.cuda(dist.get_rank())
        data_time.update(time.time()-end)
        
        output = model(data)
        loss = criterion(output, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        acc, _ = accuracy(output, target, topk=(1,5))
        
        rd_train_loss = reduce_tensor(loss.data, dist.get_world_size())
        
        train_loss.update(rd_train_loss.item(), data.shape[0])
        train_acc.update(acc[0].item(), data.size(0))
        iter_time.update(time.time()-end)
        end = time.time()
        
        if i%10 == 0 and dist.get_rank() == 0:
            print(f"[{epoch+1}/{args.epoch}] [{i+1}/{len(train_loader)}] Train Loss : {train_loss.avg:.4f} Acc(%) : {train_acc.avg:.4f} \
            Iter Time : {iter_time.avg:.4f} Data Time : {data_time.avg:.4f}")
    
    if dist.get_rank() == 0:
        train_logger.write([epoch, train_loss.avg, train_acc.avg, iter_time.avg, data_time.avg])

def test(model,criterion,test_loader,test_logger,epoch, args):
    model.eval()

    test_loss = AverageMeter()
    test_acc = AverageMeter()

    with torch.no_grad():
        for i, (data,target) in enumerate(test_loader):
            data,target = data.cuda(dist.get_rank()),target.cuda(dist.get_rank())

            output = model(data)
            loss = criterion(output,target)

            acc, _ = accuracy(output,target,topk=(1,5))
            test_loss.update(loss.item(), data.size(0))
            test_acc.update(acc[0].item(), data.size(0))

            if dist.get_rank() == 0:
                print(f"[{i+1}/{len(test_loader)}] Evaluated")
    
    if dist.get_rank() == 0:
        print(f"Epoch : [{epoch+1}/{args.epoch}] Accuracy : {test_acc.avg:.4f}")
        test_logger.write([epoch, test_loss.avg, test_acc.avg])
        
def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

    
    
def init_process(rank, world_size, backend='nccl'):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    

if __name__ == '__main__':
    args.world_size = torch.cuda.device_count()
    mp.spawn(main, nprocs = args.world_size, args = (args,))
    
    