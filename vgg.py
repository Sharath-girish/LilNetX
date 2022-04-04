from typing import Union, List, Dict, Any, cast
from numpy.lib.arraysetops import isin

import torch
import numpy as np
import torch.nn as nn
from torch.nn.modules.module import Module
from torch import Tensor
from layers import Conv2d, Linear, ConvDecoder, DenseDecoder

class VGG(nn.Module):
    def __init__(
        self, features: nn.Module, weight_decoders: Dict[str, nn.Module], bias_decoders: Dict[str, nn.Module], var:float, num_classes: int = 1000, 
        init_weights: bool = True, dropout: float = 0.5, mode: str = 'fan_out', vanilla: bool = False, boundary: float = 3.0,
    ) -> None:
        super().__init__()
        self.features = features
        self.mode = mode
        self.vanilla = vanilla
        self.boundary = boundary
        self.var = var
        self.classifier = nn.Sequential(
            Linear(512, 256, weight_decoder=weight_decoders['dense1'], bias_decoder=bias_decoders['dense1']),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            Linear(256, 256, weight_decoder=weight_decoders['dense2'], bias_decoder=bias_decoders['dense2']),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            Linear(256, num_classes, weight_decoder=weight_decoders['dense3'], bias_decoder=bias_decoders['dense3']),
        )
        self.weight_decoders = weight_decoders
        self.bias_decoders = bias_decoders
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                if not self.vanilla:
                    fan = nn.init._calculate_correct_fan(m.weight, mode=self.mode)
                    mult = 1
                    boundary = (np.sqrt(24.0/(self.var*fan/mult)+1)-1)/2.0
                    boundary = boundary/5 if isinstance(m,Linear) else boundary
                    nn.init.uniform_(m.weight,-boundary,boundary)
                else:
                    nn.init.kaiming_normal_(m.weight, mode=self.mode, nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def apply_straight_through(self, use_straight_through=False) -> None:
        for m in self.modules():
            if isinstance(m, Conv2d) or isinstance(m, Linear):
                m.use_straight_through = use_straight_through


    def get_weights(self) -> List[Tensor]:
        weights = {}
        for m in self.modules():
            if isinstance(m, Conv2d):
                weight = m.weight
                weight_reshaped = weight.reshape(weight.size(0)*weight.size(1),weight.size(2)*weight.size(3))
                if 'conv3x3' not in weights:
                    weights['conv3x3'] = weight_reshaped
                else:
                    weights['conv3x3'] = torch.cat((weights['conv3x3'],weight_reshaped))
        counter = 1
        for m in list(self.classifier.children()):
            if isinstance(m,Linear):
                weight = m.weight
                weights['dense'+str(counter)] = weight.reshape(weight.size(0)*weight.size(1),1)
                counter += 1
        assert counter == 4

        return weights



def make_layers(cfg: List[Union[str, int]], weight_decoders: Dict[str, nn.Module], bias_decoders: Dict[str, nn.Module], \
                batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = Conv2d(in_channels, v, kernel_size=3, weight_decoder=weight_decoders['conv3x3'],
                            bias_decoder=bias_decoders['conv3x3'], padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _vgg(cfg: str, batch_norm: bool, init_type: str = 'random', vanilla: bool = False, no_shift: bool = False, \
            compress_bias: bool = False, boundary:float = 3.0, **kwargs: Any) -> VGG:
    max_fan = 512
    var = 24.0/max_fan/((2*boundary+1)**2-1)

    weight_decoders = {}
    bias_decoders = {}
    weight_decoders['conv3x3'] = ConvDecoder(9,init_type, np.sqrt(var/3), no_shift) if not vanilla else nn.Identity()
    weight_decoders['dense1'] = DenseDecoder(init_type, np.sqrt(var), no_shift) if not vanilla else nn.Identity()
    weight_decoders['dense2'] = DenseDecoder(init_type, np.sqrt(var), no_shift) if not vanilla else nn.Identity()
    weight_decoders['dense3'] = DenseDecoder(init_type, np.sqrt(var), no_shift) if not vanilla else nn.Identity()
    groups = ['conv3x3', 'dense1', 'dense2', 'dense3']
    for group in groups:
        bias_decoders[group] =  DenseDecoder(init_type, np.sqrt(var), no_shift) if compress_bias and not vanilla \
                                else nn.Identity()
    model = VGG(make_layers(cfgs[cfg], weight_decoders, bias_decoders, batch_norm=batch_norm), weight_decoders, \
                bias_decoders, var, vanilla = vanilla, boundary = boundary, **kwargs)
    return model


def vgg11(**kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11", "A", False, **kwargs)


def vgg11_bn(**kwargs: Any) -> VGG:
    r"""VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg11_bn", "A", True, **kwargs)


def vgg13(**kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13", "B", False, **kwargs)


def vgg13_bn(**kwargs: Any) -> VGG:
    r"""VGG 13-layer model (configuration "B") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg13_bn", "B", True, **kwargs)


def vgg16(**kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("D", False, **kwargs)


def vgg16_bn(**kwargs: Any) -> VGG:
    r"""VGG 16-layer model (configuration "D") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("D", True,  **kwargs)


def vgg19(**kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration "E")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19", "E", False,**kwargs)


def vgg19_bn(**kwargs: Any) -> VGG:
    r"""VGG 19-layer model (configuration 'E') with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg("vgg19_bn", "E", True, **kwargs)

if __name__ == '__main__':
    model = vgg16_bn(no_shift=True, compress_bias=False,boundary = 3, num_classes = 10)