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
    
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate):
        super(_DenseLayer, self).__init__()
        
        self.add_module('conv0', nn.Conv2d(num_input_features, 4 * growth_rate, kernel_size=3, padding=1)),
        self.add_module('norm0', nn.BatchNorm2d(4 * growth_rate)),
        self.add_module('relu0', nn.ReLU(inplace=True)),
        
        self.add_module('conv1', nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1)),
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


class AntCNN(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 3 or 4 ints) - how many layers in each pooling block
        num_classes (int) - number of classification classes
        small_inputs (bool) - set to True if images are 32x32. Otherwise assumes images are larger.
    """
    def __init__(self, growth_rate=4, block_config=[4, 4, 4], num_classes=7):

        super(AntCNN, self).__init__()
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

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
