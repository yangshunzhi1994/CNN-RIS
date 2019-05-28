import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch.autograd import Variable


def _bn_function_factory(conv, norm, relu):
    def bn_function(*inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = relu(norm(conv(concated_features)))
        return bottleneck_output

    return bn_function


class LearnedGroupConv(nn.Module):
    global_progress = 0.0
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, condense_factor=None, dropout_rate=0.):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.condense_factor = condense_factor
        self.groups = groups
        self.dropout_rate = dropout_rate

        # Check if given configs are valid
        assert self.in_channels % self.groups == 0, "group value is not divisible by input channels"
        assert self.in_channels % self.condense_factor == 0, "condensation factor is not divisible by input channels"
        assert self.out_channels % self.groups == 0, "group value is not divisible by output channels"

        self.conv = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=kernel_size,
                              stride=stride, padding=padding, dilation=dilation, groups=1, bias=True)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        if self.dropout_rate > 0:
            self.dropout = nn.Dropout(self.dropout_rate, inplace=False)
        # register conv buffers
        self.register_buffer('_count', torch.zeros(1))
        self.register_buffer('_stage', torch.zeros(1))
        self.register_buffer('_mask', torch.ones(self.conv.weight.size()))

    def forward(self, x):
        # To mask the output
        weight = self.conv.weight * self.mask
        weight_bias = self.conv.bias
        out = F.conv2d(input=x, weight=weight, bias=weight_bias, stride=self.conv.stride, 
                            padding=self.conv.padding, dilation=self.conv.dilation, groups=1)
        ## Dropping here ##
        self.check_if_drop()
        
        out = self.batch_norm(out)
        out = self.relu(out)
        if self.dropout_rate > 0:
            out = self.dropout(out)
        return out

    """
    Paper: Sec 3.1: Condensation procedure: number of epochs for each condensing stage: M/2(C-1)
    Paper: Sec 3.1: Condensation factor: allow each group to select R/C of inputs.
    - During training a fraction of (C−1)/C connections are removed after each of the C-1 condensing stages
    - we remove columns in Fg (by zeroing them out) if their L1-norm is small compared to the L1-norm of other columns.
    """
    def check_if_drop(self):
        current_progress = LearnedGroupConv.global_progress
        delta = 0
        # Get current stage
        for i in range(self.condense_factor - 1):   # 3 condensation stages
            if current_progress * 2 < (i + 1) / (self.condense_factor - 1):
                stage = i
                break
        else:
            stage = self.condense_factor - 1
        # Check for actual dropping
        if not self.reach_stage(stage):
            self.stage = stage
            delta = self.in_channels // self.condense_factor
            print(delta)
        if delta > 0:
            self.drop(delta)
        return

    def drop(self, delta):
        weight = self.conv.weight * self.mask
        # Sum up all kernels
        print(weight.size())
        assert weight.size()[-1] == 1
        weight = weight.abs().squeeze()
        assert weight.size()[0] == self.out_channels
        assert weight.size()[1] == self.in_channels
        d_out = self.out_channels // self.groups
        print(d_out.size())
        # Shuffle weights
        weight = weight.view(d_out, self.groups, self.in_channels)
        print(weight.size())

        weight = weight.transpose(0, 1).contiguous()
        print(weight.size())

        weight = weight.view(self.out_channels, self.in_channels)
        print(weight.size())
        # Sort and drop
        for i in range(self.groups):
            wi = weight[i * d_out:(i + 1) * d_out, :]
            # Take corresponding delta index
            di = wi.sum(0).sort()[1][self.count:self.count + delta]
            for d in di.data:
                self._mask[i::self.groups, d, :, :].fill_(0)
        self.count = self.count + delta

    def reach_stage(self, stage):
        return (self._stage >= stage).all()

    @property
    def count(self):
        return int(self._count[0])

    @count.setter
    def count(self, val):
        self._count.fill_(val)

    @property
    def stage(self):
        return int(self._stage[0])

    @stage.setter
    def stage(self, val):
        self._stage.fill_(val)

    @property
    def mask(self):
        return Variable(self._mask)
    
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate):
        super(_DenseLayer, self).__init__()
        
        self.add_module('conv0', LearnedGroupConv(num_input_features, 4 * growth_rate, kernel_size=3, padding=1, groups=4, condense_factor=4)),
        self.add_module('norm0', nn.BatchNorm2d(4 * growth_rate)),
        self.add_module('relu0', nn.ReLU(inplace=True)),
        
        self.add_module('conv1', LearnedGroupConv(4 * growth_rate, growth_rate, kernel_size=3, padding=1, groups=8, condense_factor=8)),
        self.add_module('norm1', nn.BatchNorm2d(growth_rate)),

    def forward(self, *prev_features):
        bn_function = _bn_function_factory(self.conv0, self.norm0, self.relu0)
        if any(prev_feature.requires_grad for prev_feature in prev_features):
            bottleneck_output = cp.checkpoint(bn_function, *prev_features)
        else:
            bottleneck_output = bn_function(*prev_features)
        new_features = self.norm1(self.conv1(bottleneck_output))
        return new_features


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class _DenseBlock(nn.Module):
    def __init__(self, nChannels, growth_rate, number):
        super(_DenseBlock, self).__init__()
        layer = _DenseLayer(nChannels, growth_rate,)
        self.add_module('denselayer'+number, layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.named_children():
            new_features = layer(*features)
            features.append(new_features)
        return torch.cat(features, 1)


class EdgeNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
    """
    def __init__(self, growth_rate=4, block_config=[2, 2, 2], num_classes=7):

        super(EdgeNet, self).__init__()
        # First convolution
        
        num_features=32
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_features, kernel_size=3, stride=1, padding=1, bias=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        dense1_1 = _DenseBlock(num_features, growth_rate*block_config[0], '1_1') 
        self.features.add_module('dense1_1', dense1_1)
        num_features = num_features + growth_rate*block_config[0]
        dense1_2 = _DenseBlock(num_features, growth_rate*block_config[0], '1_2')  
        self.features.add_module('dense1_2', dense1_2)
        num_features = num_features + growth_rate*block_config[0]
        dense1_3 = _DenseBlock(num_features, growth_rate*block_config[0], '1_3') 
        self.features.add_module('dense1_3', dense1_3)
        num_features = num_features + growth_rate*block_config[0]
        dense1_4 = _DenseBlock(num_features, growth_rate*block_config[0], '1_4') 
        self.features.add_module('dense1_4', dense1_4)
        num_features = num_features + growth_rate*block_config[0]
        trans1 = _Transition(num_features, num_features)
        self.features.add_module('trans1', trans1)

        dense2_1 = _DenseBlock(num_features, growth_rate*block_config[1], '2_1') 
        self.features.add_module('dense2_1', dense2_1)
        num_features = num_features + growth_rate*block_config[1]
        dense2_2 = _DenseBlock(num_features, growth_rate*block_config[1], '2_2')
        self.features.add_module('dense2_2', dense2_2)
        num_features = num_features + growth_rate*block_config[1]
        dense2_3 = _DenseBlock(num_features, growth_rate*block_config[1], '2_3') 
        self.features.add_module('dense2_3', dense2_3)
        num_features = num_features + growth_rate*block_config[1]
        dense2_4 = _DenseBlock(num_features, growth_rate*block_config[1], '2_4') 
        self.features.add_module('dense2_4', dense2_4)
        num_features = num_features + growth_rate*block_config[1]
        trans2 = _Transition(num_features, num_features)
        self.features.add_module('trans2', trans2)   
        
        dense3_1 = _DenseBlock(num_features, growth_rate*block_config[2], '3_1') 
        self.features.add_module('dense3_1', dense3_1)
        num_features = num_features + growth_rate*block_config[2]
        dense3_2 = _DenseBlock(num_features, growth_rate*block_config[2], '3_2')
        self.features.add_module('dense3_2', dense3_2)
        num_features = num_features + growth_rate*block_config[2]
        dense3_3 = _DenseBlock(num_features, growth_rate*block_config[2], '3_3') 
        self.features.add_module('dense3_3', dense3_3)
        num_features = num_features + growth_rate*block_config[2]
        dense3_4 = _DenseBlock(num_features, growth_rate*block_config[2], '3_4') 
        self.features.add_module('dense3_4', dense3_4)
        num_features = num_features + growth_rate*block_config[2]
        dense3_5 = _DenseBlock(num_features, growth_rate*block_config[2], '3_5') 
        self.features.add_module('dense3_5', dense3_5)
        num_features = num_features + growth_rate*block_config[2]
        dense3_6 = _DenseBlock(num_features, growth_rate*block_config[2], '3_6') 
        self.features.add_module('dense3_6', dense3_6)
        num_features = num_features + growth_rate*block_config[2]
        dense3_7 = _DenseBlock(num_features, growth_rate*block_config[2], '3_7') 
        self.features.add_module('dense3_7', dense3_7)
        num_features = num_features + growth_rate*block_config[2]
        
        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

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
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=5).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    