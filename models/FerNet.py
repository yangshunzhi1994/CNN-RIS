import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import init
from torch._jit_internal import weak_module, weak_script_method
from torch.nn.parameter import Parameter
import math
from torch.autograd import Function

class Maxout(Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        x = input
        max_out=4    #Maxout Parameter
        kernels = x.shape[1]  # to get how many kernels/output
        feature_maps = int(kernels / max_out)
        out_shape = (x.shape[0], feature_maps, max_out, x.shape[2], x.shape[3])
        x= x.view(out_shape)
        y, indices = torch.max(x[:, :, :], 2)
        ctx.save_for_backward(input)
        ctx.indices=indices
        ctx.max_out=max_out
        return y

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        input1,indices,max_out= ctx.saved_variables[0],Variable(ctx.indices),ctx.max_out
        input=input1.clone()
        for i in range(max_out):
            a0=indices==i
            input[:,i:input.data.shape[1]:max_out]=a0.float()*grad_output
      

        return input

class FerNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
    """
    def __init__(self, num_classes=7):

        super(FerNet, self).__init__()
        
        self.Conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.Conv2 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=5, stride=1, padding=2, bias=True),
            nn.Dropout(0.5)
        )

        self.batchNorm1 = nn.BatchNorm2d(64)

        self.Conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2, bias=True),
            nn.Dropout(0.5)
        )

        self.batchNorm2 = nn.BatchNorm2d(128)  
        self.Maxout = Maxout.apply    
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.IP1 = nn.Sequential(
            nn.Linear(14112, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.IP2 = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(1024, num_classes)
        
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
        x_Conv1 = self.Conv1(x)
        x_Conv2 = self.Conv2(x_Conv1)
        x_Conv3 = torch.cat((x_Conv1, x_Conv2), 1)
        x_Conv3 = F.relu(self.batchNorm1(x_Conv3), inplace=True)
        x_Conv3 = self.Conv3(x_Conv3)
        x = torch.cat((x_Conv1, x_Conv2, x_Conv3), 1)

        x = self.batchNorm2(x)

        x = self.Maxout(x)

        x = self.pool1(x)

        x = x.view(x.size(0), -1)
        
        x = self.IP1(x)
        x = self.IP2(x)
        out = self.classifier(x)
        return out

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
