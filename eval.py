
import torch
import torch.nn.functional as F

import time
import numpy as np
from copy import deepcopy

import utils


def evaluate(data_loader, model, criterion, epoch):
    model.eval()

    running_loss = 0
    total = 0
    predictions = 0
    
    with torch.no_grad():
        print('eval on')

        for i, (input, target) in enumerate(data_loader):

            input = input.to("cuda")
            target = target.long().to("cuda")

            output = model(input)
            _, predicted = output.max(1)

            loss = criterion(output, target)
            running_loss += loss.item()
            total += target.size(0)

            predictions += np.sum(predicted.cpu().numpy() == target.cpu().numpy())/target.size(0)

        print('eval done')
        print(f'accuracy : {predictions/len(data_loader)}')
        return predictions/len(data_loader)