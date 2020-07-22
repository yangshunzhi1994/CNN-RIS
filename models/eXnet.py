import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
    
class RouteA(nn.Module):

    def __init__(self, in_channel, out_channel):

        super().__init__()
        self.CBR_1x1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.CBR_3x3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        
        x = self.CBR_1x1(x)
        x = self.CBR_3x3(x)

        return x


class RouteB(nn.Module):

    def __init__(self, in_channel, out_channel):

        super().__init__()
        self.pool_3x3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        
        self.CBR_1x1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.CBR_3x3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):

        x = self.pool_3x3(x)
        x = self.CBR_1x1(x)
        x = self.CBR_3x3(x)

        return x


class ParaFeat(nn.Module):

    def __init__(self, in_channel, out_channel):

        super().__init__()
        self.RouteA = RouteA(in_channel, out_channel)
        self.RouteB = RouteB(in_channel, out_channel)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    
    def forward(self, x):

        x_RouteA = self.RouteA(x)
        x_RouteB = self.RouteB(x)
        x = torch.cat([x_RouteA, x_RouteB], 1)
        x = self.pool(x)

        return x

class Final_Feature(nn.Module):

    def __init__(self, in_channel, out_channel):

        super().__init__()
        self.CBR_1x1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.pool_1_2x2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)

        self.CBR_3x3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True)
        )

        self.pool_2_2x2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
    
    def forward(self, x):

        x = self.CBR_1x1(x)
        x = self.pool_1_2x2(x)
        x = self.CBR_3x3(x)
        x = self.pool_2_2x2(x)
        x = F.avg_pool2d(x, kernel_size=3).view(x.size(0), -1)

        return x

class eXnet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
    """
    def __init__(self, num_classes=7):

        super(eXnet, self).__init__()
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.Conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.batchNorm1 = nn.BatchNorm2d(64)
        self.ParaFeat1 = ParaFeat(64, 128)
        self.ParaFeat2 = ParaFeat(256, 256)
        self.Final_Feature = Final_Feature(512, 512)
        self.fc = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_classes)
        
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
        x = self.Conv1(x)
        x = self.Conv2(x)
        x = self.pool1(x)
        x = self.batchNorm1(x)
        x = self.ParaFeat1(x)
        x = self.ParaFeat2(x)
        x = self.Final_Feature(x)
        x = self.fc(x)
        out = self.classifier(x)
        return out

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
