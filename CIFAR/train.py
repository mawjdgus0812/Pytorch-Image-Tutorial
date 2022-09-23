import torch
import torch.nn.functional as F

import numpy as np

import time
import utils


def train(args, data_loader, model, optimizer, criterion, epoch):
    model.train()

    running_loss = 0; total = 0;
    predictions = 0

    print('train on')

    for i, (input, target) in enumerate(data_loader):
        # input, traget
        input = input.to("cuda")
        target = target.long().to("cuda")

        # zero the parameter gradient
        optimizer.zero_grad()

        # forward + backward + optimize
        output = model(input)
        _, predicted = output.max(1)

        
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        total += target.size(0)

        predictions += np.sum(predicted.cpu().numpy() == target.cpu().numpy())/target.size(0)

        if i % 10==0:
            print(f'[{epoch+1}/{args.epoch}], [{i + 1}/{len(data_loader)}] loss : {running_loss / len(data_loader)}')

    print('Finished Training')
    print(f'accuracy : {predictions/len(data_loader)}')
