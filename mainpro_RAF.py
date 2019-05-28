'''Train RAF with PyTorch.'''
# 10 crop for data enhancement
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import transforms as transforms
import numpy as np
import os
import time
import argparse
import utils
from data.RAF import RAF
from torch.autograd import Variable
from models import *

parser = argparse.ArgumentParser(description='PyTorch RAF CNN Training')
parser.add_argument('--model', type=str, default='EdgeNet', help='CNN architecture')
parser.add_argument('--dataset', type=str, default='/home/ysz/Mask_RCNN/data/pytoch/Expression10/models/RAF', help='CNN architecture')
parser.add_argument('--train_bs', default=128, type=int, help='learning rate')
parser.add_argument('--test_bs', default=4, type=int, help='learning rate')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--resume', default=True, type=int, help='resume from checkpoint')
opt = parser.parse_args()

use_cuda = torch.cuda.is_available()
best_PrivateTest_acc = 0  # best PrivateTest accuracy
best_PrivateTest_acc_epoch = 0
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

learning_rate_decay_start = 80  # 50
learning_rate_decay_every = 5 # 5
learning_rate_decay_rate = 0.9 # 0.9

cut_size = 44
total_epoch = 800

total_prediction_fps = 0 
total_prediction_n = 0

path = os.path.join(opt.dataset + '_' + opt.model)

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = RAF(split = 'Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.train_bs, shuffle=True, num_workers=1)

PrivateTestset = RAF(split = 'PrivateTest', transform=transform_test)
PrivateTestloader = torch.utils.data.DataLoader(PrivateTestset, batch_size=opt.test_bs, shuffle=False, num_workers=1)

# Model
if opt.model == 'VGG19':
    net = VGG('VGG19')
elif opt.model  == 'Resnet18':
    net = ResNet18()
elif opt.model  == 'EdgeNet':
    print ("This is EdgeNet ")
    net = EdgeNet()


if opt.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(path), 'Error: no checkpoint directory found!'
    Private_checkpoint = torch.load(os.path.join(path,'PrivateTest_model.t7'))
    best_PrivateTest_acc = Private_checkpoint['best_PrivateTest_acc']
    best_PrivateTest_acc_epoch = Private_checkpoint['best_PrivateTest_acc_epoch']
    
    print ('best_PrivateTest_acc is '+ str(best_PrivateTest_acc))
    net.load_state_dict(Private_checkpoint['net'])
    start_epoch = Private_checkpoint['best_PrivateTest_acc_epoch'] + 1
    
else:
    print('==> Building model..')

if use_cuda:
    net.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=opt.lr, momentum=0.9, weight_decay=5e-4)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    global Train_acc
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if epoch > learning_rate_decay_start and learning_rate_decay_start >= 0:
        frac = (epoch - learning_rate_decay_start) // learning_rate_decay_every
        decay_factor = learning_rate_decay_rate ** frac
        current_lr = opt.lr * decay_factor
        utils.set_lr(optimizer, current_lr)  # set the decayed rate
    else:
        current_lr = opt.lr
    print('learning_rate: %s' % str(current_lr))

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        utils.clip_gradient(optimizer, 0.1)
        optimizer.step()
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (train_loss/(batch_idx+1), 100.*float(correct)/float(total), correct, total))

    Train_acc = 100.*float(correct)/float(total)

def PrivateTest(epoch):
    global PrivateTest_acc
    global best_PrivateTest_acc
    global best_PrivateTest_acc_epoch
    global total_prediction_fps
    global total_prediction_n
    net.eval()
    PrivateTest_loss = 0
    correct = 0
    total = 0
    t_prediction = 0
    for batch_idx, (inputs, targets) in enumerate(PrivateTestloader):
        t = time.time()
        test_bs, ncrops, c, h, w = np.shape(inputs)
        inputs = inputs.view(-1, c, h, w)
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)
        outputs_avg = outputs.view(test_bs, ncrops, -1).mean(1)  # avg over crops
        _, predicted = torch.max(outputs_avg.data, 1)
        t_prediction += (time.time() - t)
        
        loss = criterion(outputs_avg, targets)
        PrivateTest_loss += loss.item()
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        utils.progress_bar(batch_idx, len(PrivateTestloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (PrivateTest_loss / (batch_idx + 1), 100. *  float(correct) / float(total), correct, total))
    total_prediction_fps = total_prediction_fps + (1 / (t_prediction / len(PrivateTestloader)))
    total_prediction_n = total_prediction_n + 1
    print('Prediction time: %.2f' % t_prediction + ', Average : %.5f/image' % (t_prediction / len(PrivateTestloader)) 
         + ', Speed : %.2fFPS' % (1 / (t_prediction / len(PrivateTestloader))))
    
    # Save checkpoint.
    PrivateTest_acc = 100.* float(correct) / float(total)
    if PrivateTest_acc > best_PrivateTest_acc:
        print('Saving..')
        print("best_PrivateTest_acc: %0.3f" % PrivateTest_acc)
        state = {
            'net': net.state_dict() if use_cuda else net,
            'best_PrivateTest_acc': PrivateTest_acc,
            'best_PrivateTest_acc_epoch': epoch,
        }
        if not os.path.isdir(path):
            os.mkdir(path)
        torch.save(state, os.path.join(path,'PrivateTest_model.t7'))
        best_PrivateTest_acc = PrivateTest_acc
        best_PrivateTest_acc_epoch = epoch

for epoch in range(start_epoch, total_epoch):
    train(epoch)
    PrivateTest(epoch)

print("best_PrivateTest_acc: %0.3f" % best_PrivateTest_acc)
print("best_PrivateTest_acc_epoch: %d" % best_PrivateTest_acc_epoch)

print("total_prediction_fps: %0.2f" % total_prediction_fps)
print("total_prediction_n: %d" % total_prediction_n)
print('Average speed: %.2f FPS' % (total_prediction_fps / total_prediction_n))
