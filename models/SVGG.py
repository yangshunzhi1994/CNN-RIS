import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SVGG(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
    """
    def __init__(self, num_classes=7):

        super(SVGG, self).__init__()
        
        self.Conv1_1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU(),
            nn.BatchNorm2d(32)
        )
        self.Conv1_2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU(),
            nn.BatchNorm2d(32)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.Conv2_1 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU(),
            nn.BatchNorm2d(64)
        )
        self.Conv2_2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU(),
            nn.BatchNorm2d(64)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)


        self.Conv3_1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU(),
            nn.BatchNorm2d(128)
        )
        self.Conv3_2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ELU(),
            nn.BatchNorm2d(128)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.IP1 = nn.Linear(3200, 64)
        self.IP2 = nn.Linear(64, 64)
        self.IP3 = nn.Linear(64, num_classes)
        
        # Initialization
        for m in self.named_parameters():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        x = self.pool1(self.Conv1_2(self.Conv1_1(x)))
        x = self.pool2(self.Conv2_2(self.Conv2_1(x)))
        x = self.pool3(self.Conv3_2(self.Conv3_1(x)))

        x = x.view(x.size(0), -1) 

        x = self.IP1(x)
        x = self.IP2(x)
        x = self.IP3(x)
        
        out = F.softmax(x,dim=1)
        return out

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
