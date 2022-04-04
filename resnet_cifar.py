from typing import List
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from layers import Conv2d, Linear, ConvDecoder, DenseDecoder
from torch import Tensor
from torch.autograd import Variable

__all__ = ['ResNet', 'resnet20', 'resnet32', 'resnet44', 'resnet56', 'resnet110', 'resnet1202']

def _weights_init(m, var, mode, vanilla):
    classname = m.__class__.__name__
    if isinstance(m, Linear) or isinstance(m, Conv2d):
        if not vanilla:
            fan = init._calculate_correct_fan(m.weight, mode=mode)
            boundary = (np.sqrt(24.0/(var*fan)+1)-1)/2.0
            init.uniform_(m.weight,-boundary,boundary)
        else:
            init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlockCifar(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, weight_decoders, bias_decoders, stride=1, option='A'):
        super(BasicBlockCifar, self).__init__()
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False,\
                            weight_decoder=weight_decoders['conv3x3'], bias_decoder=bias_decoders['conv3x3'])
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, \
                            weight_decoder=weight_decoders['conv3x3'], bias_decoder=bias_decoders['conv3x3'])
        self.bn2 = nn.BatchNorm2d(planes)
        self.stride = stride
        self.option = option
        self.in_planes = in_planes
        self.planes = planes
        self.relu = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False, \
                                weight_decoder=weight_decoders['conv1x1'], bias_decoder=bias_decoders['conv1x1']),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNetCifar(nn.Module):
    def __init__(self, block, num_blocks, width=1, option='A', num_classes=10, \
                 init_type='random', compress_bias = False, vanilla=False,\
                 mode='fan_out', boundary=10, no_shift=False):
        super(ResNetCifar, self).__init__()

        self.in_planes = 16*width

        weight_decoders = {}
        bias_decoders = {}
        max_fan = 64*width
        var = 24.0/max_fan/((2*boundary+1)**2-1)
        weight_decoders['conv3x3'] = ConvDecoder(9,init_type, np.sqrt(var), no_shift) if not vanilla else nn.Identity()
        weight_decoders['dense'] = DenseDecoder(init_type, np.sqrt(var), no_shift) if not vanilla else nn.Identity()
        groups = ['conv3x3', 'dense']
        for group in groups:
            bias_decoders[group] =  DenseDecoder(init_type, np.sqrt(var), no_shift) if compress_bias and not vanilla \
                                     else nn.Identity()

        self.weight_decoders = weight_decoders
        self.bias_decoders = bias_decoders

        self.conv1 = Conv2d(3, 16*width, kernel_size=3, stride=1, padding=1, bias=False, \
                            weight_decoder=weight_decoders['conv3x3'], bias_decoder=bias_decoders['conv3x3'])
        self.bn1 = nn.BatchNorm2d(16*width)
        self.layer1 = self._make_layer(block, 16*width, num_blocks[0], option, stride=1)
        self.layer2 = self._make_layer(block, 32*width, num_blocks[1], option, stride=2)
        self.layer3 = self._make_layer(block, 64*width, num_blocks[2], option, stride=2)
        self.fc = Linear(64*width, num_classes, weight_decoders['dense'], bias_decoders['dense'],\
                         compress_bias = compress_bias)
        # self.avg_pool2d = nn.AvgPool2d(8)

        self.apply(lambda m: _weights_init(m, var, mode, vanilla))

    def _make_layer(self, block, planes, num_blocks, option, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, self.weight_decoders, self.bias_decoders, stride, option))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        # out = self.avg_pool2d(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

    def apply_straight_through(self, use_straight_through=False) -> None:
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                m.use_straight_through = use_straight_through

    def apply_compress_bias(self, compress_bias=False) -> None:
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                m.compress_bias = compress_bias

    def get_weights(self) -> List[Tensor]:
        weights = {}
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                weight = m.weight
                group_name = self.get_group_name(m.weight)
                if group_name == 'dense':
                    weight_reshaped = weight.reshape(weight.size(0)*weight.size(1),1)
                else:
                    weight_reshaped = weight.reshape(weight.size(0)*weight.size(1),weight.size(2)*weight.size(3))
                if group_name in weights:
                    weights[group_name] = torch.cat((weights[group_name],weight_reshaped))
                else:
                    weights[group_name] = weight_reshaped
        return weights

    def get_biases(self) -> List[Tensor]:
        biases = {}
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                bias = m.bias
                group_name = self.get_group_name(m.weight)
                if group_name in biases and bias is not None:
                    biases[group_name] = torch.cat((biases[group_name],bias))
                elif bias is not None:
                    biases[group_name] = bias
        return biases

    def get_group_name(self, param):
        if param.dim()==2:
            return 'dense'
        else:
            h = param.size(2)
            w = param.size(3)
            return f'conv{h}x{w}'

def _resnet_cifar(
    block,
    layers,
    **kwargs
) -> ResNetCifar:
    model = ResNetCifar(block, layers, **kwargs)
    return model

