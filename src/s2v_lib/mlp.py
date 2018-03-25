from __future__ import print_function

import os
import sys
import numpy as np
import torch
import random
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from pytorch_util import weights_init

class MLPRegression(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(MLPRegression, self).__init__()

        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, 1)

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        pred = self.h2_weights(h1)
        
        if y is not None:
            y = Variable(y)
            mse = F.mse_loss(pred, y)
            mae = F.l1_loss(pred, y)
            return pred, mae, mse
        else:
            return pred

class MLPClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_class):
        super(MLPClassifier, self).__init__()

        # KM - Commented out
        self.h1_weights = nn.Linear(input_size, hidden_size)
        self.h2_weights = nn.Linear(hidden_size, num_class)
        # KM - Commented out
        
        # KM - added
        # print("Keegan here!")
        #self.h1_weights = nn.Linear(input_size, hidden_size)
        #self.h2_weights = nn.Linear(hidden_size, hidden_size)
        #self.h3_weights = nn.Linear(hidden_size, hidden_size)
        #self.h4_weights = nn.Linear(hidden_size, hidden_size)
        #self.h5_weights = nn.Linear(hidden_size, num_class)
        #self.dropout = nn.Dropout(0.5)
        # KM - added

        weights_init(self)

    def forward(self, x, y = None):
        h1 = self.h1_weights(x)
        h1 = F.relu(h1)

        # KM - added
        #h1 = self.dropout(h1)

        #h2 = self.h2_weights(h1)
        #h2 = F.relu(h2)
        #h2 = self.dropout(h2)

        #h3 = self.h2_weights(h2)
        #h3 = F.relu(h3)
        #h3 = self.dropout(h3)

        #h4 = self.h2_weights(h3)
        #h4 = F.relu(h4)
        #h4 = self.dropout(h4)

        #logits = self.h5_weights(h4)
        #logits = F.log_softmax(logits, dim=1)
        # KM - added

        # KM - Commented out
        logits = self.h2_weights(h1)
        logits = F.log_softmax(logits, dim=1)
        # KM - Commented out

        if y is not None:
            y = Variable(y)
            loss = F.nll_loss(logits, y)

            pred = logits.data.max(1, keepdim=True)[1]
            acc = pred.eq(y.data.view_as(pred)).cpu().sum() / float(y.size()[0])
            return logits, loss, acc
        else:
            return logits