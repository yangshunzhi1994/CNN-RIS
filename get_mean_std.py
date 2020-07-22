'''Train Fer2013 with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import transforms as transforms
import numpy as np
import os
import time
import argparse
import utils
from data.RAF import RAF
from data.fer import FER2013
from torch.autograd import Variable
from models import *
from thop import profile
from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='PyTorch Fer2013 CNN Training')
parser.add_argument('--model', type=str, default='AntCNN', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='models/FER2013', help='CNN architecture')
parser.add_argument('--train_bs', default=128, type=int, help='learning rate')
parser.add_argument('--test_bs', default=64, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', default=False, type=int, help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

total_epoch = 800

total_prediction_fps = 0 
total_prediction_n = 0

path = os.path.join(opt.dataset + '_' + opt.model)
writer = SummaryWriter(log_dir=os.path.join(opt.dataset + '_' + opt.model))

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(44),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    utils.Cutout(n_holes=1, length=10),
])

transform_test = transforms.Compose([
    transforms.TenCrop(44),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

if opt.dataset  == 'models/FER2013':
    print('This is FER2013..')
    trainset = FER2013(split = 'Training', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_bs, 
        shuffle=True, num_workers=1)
    PrivateTestset = FER2013(split = 'PrivateTest', transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.test_bs,
        shuffle=False, num_workers=1)
else:
    print('This is RAF..')
    trainset = RAF(split = 'Training', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_bs, 
        shuffle=True, num_workers=1)

    PrivateTestset = RAF(split = 'PrivateTest', transform=transform_test)
    PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.test_bs,
        shuffle=False, num_workers=1)



train_mean=0 
train_std=0
epoch_mean=0 
epoch_std=0

for epoch in range(1, 2):
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets_a, targets_b, lam = utils.mixup_data(inputs, targets, 0.6, True)
        train_mean += np.mean(inputs.numpy(), axis=(0,2,3))
        train_std += np.std(inputs.numpy(), axis=(0,2,3))
        mean = train_mean/(batch_idx+1)
        std = train_std/(batch_idx+1)    
    train_mean=0 
    train_std=0
    epoch_mean += mean
    epoch_std += std
print('------train---------')
print (epoch_mean/epoch, epoch_std/epoch)


test_mean=0 
test_std=0
epoch_mean=0 
epoch_std=0

for epoch in range(1, 2):
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        test_mean += np.mean(inputs.numpy(), axis=(0,1,3,4))
        test_std += np.std(inputs.numpy(), axis=(0,1,3,4))
        mean = test_mean/(batch_idx+1)
        std = test_std/(batch_idx+1)
    test_mean=0 
    test_std=0
    epoch_mean += mean
    epoch_std += std
print('------test---------')
print (epoch_mean/epoch, epoch_std/epoch)

