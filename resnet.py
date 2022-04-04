import torch
import os
import numpy as np
from torch import Tensor
from layers import Conv2d, Linear, ConvDecoder, DenseDecoder
import torch.nn as nn
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer('blur_filter', filt)

    def forward(self, x):
        blurred = F.conv2d(x, self.blur_filter, stride=1, padding=(1, 1),
                           groups=self.conv.in_channels, bias=None)
        return self.conv.forward(blurred)

def conv3x3(in_planes: int, out_planes: int, weight_decoder, bias_decoder, 
            stride: int = 1, groups: int = 1, dilation: int = 1, padding: int=1) -> Conv2d:
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=padding, groups=groups, bias=False, dilation=dilation,
                     weight_decoder=weight_decoder, bias_decoder=bias_decoder)


def conv1x1(in_planes: int, out_planes: int, weight_decoder, bias_decoder,
            stride: int = 1) -> Conv2d:
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False,
                     weight_decoder=weight_decoder, bias_decoder=bias_decoder)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        weight_decoders: dict,
        bias_decoders: dict,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, weight_decoders['conv3x3'], bias_decoders['conv3x3'], stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, weight_decoders['conv3x3'], bias_decoders['conv3x3'])
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        weight_decoders: dict,
        bias_decoders: dict,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, weight_decoders['conv1x1'], bias_decoders['conv1x1'])
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, weight_decoders['conv3x3'], bias_decoders['conv3x3'], 
                             stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, weight_decoders['conv1x1'], 
                             bias_decoders['conv1x1'])
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        init_type: str = 'random',
        vanilla: bool = False,
        large: bool = True,
        mode: str = 'fan_out',
        boundary = 10,
        compress_bias: bool = False,
        no_shift: bool = False,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.block = block
        self.groups = groups

        self.inplanes = 64
        self.dilation = 1
        max_fan = 512 * block.expansion
        max_fan = 4608
        var = 24.0/max_fan/(2*boundary)**2
        # max_fan = 4608
        # if block == BasicBlock:
        #     max_fan = {'conv7x7': 3136, 'conv1x1': 512, 'conv3x3': 4608, 'fc':1000} if mode=='fan_out' else \
        #             {'conv7x7': 147, 'conv1x1': 256, 'conv3x3': 4608, 'fc': 512}
        # var = 1/(3*boundary)/(3*boundary)

        weight_decoders = {}
        bias_decoders = {}
        if large:
            weight_decoders['conv7x7'] = ConvDecoder(49,'random', np.sqrt(1/7), no_shift) if not vanilla else nn.Identity()
        weight_decoders['conv1x1'] = DenseDecoder('random', np.sqrt(1), no_shift) if not vanilla else nn.Identity()
        weight_decoders['conv3x3'] = ConvDecoder(9,'random', np.sqrt(1/3), no_shift) if not vanilla else nn.Identity()
        weight_decoders['dense'] = DenseDecoder('random', np.sqrt(1), no_shift) if not vanilla else nn.Identity()
        groups = ['conv7x7'] if large else []
        groups += ['conv1x1']
        groups += ['conv3x3', 'dense']
        for group in groups:
            bias_decoders[group] =  DenseDecoder('random', np.sqrt(1), no_shift) if compress_bias and not vanilla \
                                    else nn.Identity()

        self.weight_decoders = weight_decoders
        self.bias_decoders = bias_decoders
        self.large = large

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.base_width = width_per_group
        if large:
            kernel_size = 7
            stride = 2
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            weight_decoder = weight_decoders['conv7x7']
            bias_decoder = bias_decoders['conv7x7']
            padding = 3
        else:
            kernel_size = 3
            stride = 1
            padding = 1
            self.maxpool = nn.Identity()
            weight_decoder = weight_decoders['conv3x3']
            bias_decoder = bias_decoders['conv3x3']
        self.conv1 = Conv2d(3, self.inplanes, kernel_size=kernel_size, stride=stride, padding=padding,
                               bias=False, weight_decoder=weight_decoder, 
                               bias_decoder=bias_decoder)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = Linear(512 * block.expansion, num_classes, weight_decoders['dense'], bias_decoders['dense'])

        max_fan = self.calc_max_fan(mode)
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                if not vanilla:
                    fan = nn.init._calculate_correct_fan(m.weight, mode=mode)
                    group = self.get_group_name(m.weight)
                    mult = round(max_fan[group]/fan)
                    # nn.init.uniform_(m.weight,mult-0.5,mult+0.5)
                    # m.weight.data.multiply_((torch.bernoulli(torch.ones_like(m.weight.data))-0.5)*2)
                    
                    if init_type == 'var':
                        boundary = (np.sqrt(24.0/(var*fan/mult)+1)-1)/2.0
                    # b = np.sqrt(24.0/fan)/2*(2*np.sqrt(max_fan/24)*boundary)
                    # (b-a)^2/12 = 2/fan


                    # std = 3*boundary*np.sqrt(2/fan)
                    # nn.init.normal_(m.weight,0.0,2*boundary/np.sqrt(12)/10)
                    nn.init.normal_(m.weight,0.0,boundary)
                else:
                    nn.init.kaiming_normal_(m.weight, mode=mode, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        if not vanilla:
            mults = {'conv7x7':7, 'conv3x3': 3, 'conv1x1': 1, 'dense': 1}
            for group in weight_decoders:
                # weight_decoders[group].reset_parameters(init_type,np.sqrt(2/max_fan[group]/mults[group]))
                weight_decoders[group].reset_parameters('random',np.sqrt(var/mults[group]))
                if compress_bias:
                    # bias_decoders[group].reset_parameters(init_type, np.sqrt(2/max_fan[group]))
                    bias_decoders[group].reset_parameters('random', np.sqrt(2/max_fan[group]))

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def calc_max_fan(self, mode):
        max_fan = {}
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                fan = nn.init._calculate_correct_fan(m.weight, mode=mode)
                group = self.get_group_name(m.weight)
                if group not in max_fan:
                    max_fan[group] = fan
                else:
                    max_fan[group] = max(fan,max_fan[group])
        return max_fan

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, self.weight_decoders['conv1x1'], \
                        self.bias_decoders['conv1x1'], stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, self.weight_decoders, self.bias_decoders, stride,
                             downsample, self.groups, self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, self.weight_decoders, self.bias_decoders, 
                                groups=self.groups,base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        # import time
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # print('Pausing at 0')
        # torch.cuda.synchronize()
        # time.sleep(20)
        x = self.layer1(x)
        # print('Pausing at 1')
        # torch.cuda.synchronize()
        # time.sleep(20)
        x = self.layer2(x)
        # print('Pausing at 2')
        # torch.cuda.synchronize()
        # time.sleep(20)
        x = self.layer3(x)
        # print('Pausing at 3')
        # torch.cuda.synchronize()
        # time.sleep(20)
        x = self.layer4(x)
        # print('Pausing at 4')
        # torch.cuda.synchronize()
        # time.sleep(20)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

    # def apply_straight_through(self, use_straight_through=False) -> None:
    #     self.conv1.use_straight_through = use_straight_through
    #     for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
    #         for block in list(layer.children()):
    #             block.conv1.use_straight_through = use_straight_through
    #             block.conv2.use_straight_through = use_straight_through
    #             if self.block == Bottleneck:
    #                 block.conv3.use_straight_through = use_straight_through
    #     self.fc.use_straight_through = use_straight_through

    # def apply_compress_bias(self, compress_bias=False) -> None:
    #     self.conv1.compress_bias = compress_bias
    #     for layer in [self.layer1, self.layer2, self.layer3, self.layer4]:
    #         for block in list(layer.children()):
    #             block.conv1.compress_bias = compress_bias
    #             block.conv2.compress_bias = compress_bias
    #             if self.block == Bottleneck:
    #                 block.conv3.compress_bias = compress_bias
    #     self.fc.compress_bias = compress_bias


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

def _resnet(
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers,  **kwargs)
    return model

