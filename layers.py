import math
import warnings

import torch
from torch import Tensor
from torch.nn.parameter import Parameter, UninitializedParameter
import torch.nn.functional as F
from torch.nn import init
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch._torch_docs import reproducibility_notes

from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from typing import Optional, List, Tuple, Union

class _ConvNd(Module):

    __constants__ = ['stride', 'padding', 'dilation', 'groups',
                     'padding_mode', 'output_padding', 'in_channels',
                     'out_channels', 'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]) -> Tensor:
        ...

    _in_channels: int
    _reversed_padding_repeated_twice: List[int]
    out_channels: int
    kernel_size: Tuple[int, ...]
    stride: Tuple[int, ...]
    padding: Union[str, Tuple[int, ...]]
    dilation: Tuple[int, ...]
    transposed: bool
    output_padding: Tuple[int, ...]
    groups: int
    padding_mode: str
    weight: Tensor
    bias: Optional[Tensor]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: Tuple[int, ...],
                 stride: Tuple[int, ...],
                 padding: Tuple[int, ...],
                 dilation: Tuple[int, ...],
                 transposed: bool,
                 output_padding: Tuple[int, ...],
                 groups: int,
                 bias: bool,
                 padding_mode: str,
                 device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(_ConvNd, self).__init__()
        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")

        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        self.padding_mode = padding_mode
        # `_reversed_padding_repeated_twice` is the padding to be passed to
        # `F.pad` if needed (e.g., for non-zero padding types that are
        # implemented as two ops: padding + conv). `F.pad` accepts paddings in
        # reverse order than the dimension.
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(kernel_size)
            if padding == 'same':
                for d, k, i in zip(dilation, kernel_size,
                                   range(len(kernel_size) - 1, -1, -1)):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad)
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(self.padding, 2)

        if transposed:
            self.weight = Parameter(torch.empty(
                (in_channels, out_channels // groups, *kernel_size), **factory_kwargs))
        else:
            self.weight = Parameter(torch.empty(
                (out_channels, in_channels // groups, *kernel_size), **factory_kwargs))
        if bias is not None:
            self.bias = Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def extra_repr(self):
        s = ('{in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        if self.padding_mode != 'zeros':
            s += ', padding_mode={padding_mode}'
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        super(_ConvNd, self).__setstate__(state)
        if not hasattr(self, 'padding_mode'):
            self.padding_mode = 'zeros'



class Conv2d(_ConvNd):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        weight_decoder,
        bias_decoder,
        use_straight_through:bool = False,
        compress_bias:bool = False,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',  # TODO: refine this type
        device=None,
        dtype=None
    ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        kernel_size_ = _pair(kernel_size)
        stride_ = _pair(stride)
        padding_ = padding if isinstance(padding, str) else _pair(padding)
        dilation_ = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size_, stride_, padding_, dilation_,
            False, _pair(0), groups, bias, padding_mode, **factory_kwargs)
        self.weight_decoder = weight_decoder
        self.bias_decoder = bias_decoder
        self.use_straight_through = use_straight_through
        self.compress_bias = compress_bias

    def _conv_forward(self, input: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        out = F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

        return out

    def forward(self, input: Tensor) -> Tensor:
        weight = StraightThrough.apply(self.weight) if self.use_straight_through else self.weight
        bias = StraightThrough.apply(self.bias) if self.use_straight_through and self.bias is not None \
                                                   and self.compress_bias else self.bias
        out = self._conv_forward(input, self.weight_decoder(weight), \
                                  self.bias_decoder(bias) if bias is not None else None)

        return out

class Linear(Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int, weight_decoder, bias_decoder, compress_bias:bool = False,
                 use_straight_through: bool = False, bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_decoder = weight_decoder
        self.bias_decoder = bias_decoder
        self.use_straight_through = use_straight_through
        self.compress_bias = compress_bias
        self.weight = Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        if bias is not None:
            self.bias = Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input: Tensor) -> Tensor:
        weight = StraightThrough.apply(self.weight) if self.use_straight_through else self.weight
        bias = StraightThrough.apply(self.bias) if self.use_straight_through and self.bias is not None \
                                                   and self.compress_bias else self.bias
        return F.linear(input, self.weight_decoder(weight), \
                                  self.bias_decoder(bias) if bias is not None else None)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

class ConvDecoder(Module):
    def __init__(
        self,
        channels: int,
        init_type: str,
        std: float,
        no_shift: bool,
        device=None,
        dtype=None
    ) -> None:
        super(ConvDecoder, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.channels = channels
        self.scale = Parameter(torch.empty((channels, channels), **factory_kwargs))
        self.shift = Parameter(torch.empty((1,channels), **factory_kwargs)) if not no_shift else 0.0

        self.no_shift = no_shift
        self.reset_parameters(init_type, std)

    def reset_parameters(self, init_type, std) -> None:
        if init_type == 'random':
            init.normal_(self.scale, std=std)
            # init.normal_(self.shift)
        else:
            init.eye_(self.scale)
        if not self.no_shift:
            init.zeros_(self.shift)

    def forward(self, input: Tensor) -> Tensor:
        # assert input.dim() == 4 and input.size(2)*input.size(3)==self.channels
        w_in = input.reshape(input.size(0)*input.size(1),input.size(2)*input.size(3)) #assume oixhw
        w_out = torch.matmul(w_in+self.shift,self.scale)
        return w_out.reshape(input.size())


class DenseDecoder(Module):
    def __init__(
        self,
        init_type: str,
        std: float,
        no_shift: bool,
        device=None,
        dtype=None
    ) -> None:
        super(DenseDecoder, self).__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.scale = Parameter(torch.empty((1), **factory_kwargs))
        self.shift = Parameter(torch.empty((1), **factory_kwargs)) if not no_shift else 0.0

        self.no_shift = no_shift
        self.reset_parameters(init_type, std)

    def reset_parameters(self, init_type, std) -> None:
        if init_type == 'random':
            init.normal_(self.scale, std=std)
        else:
            init.ones_(self.scale)
        if not self.no_shift:
            init.zeros_(self.shift)

    def forward(self, input: Tensor) -> Tensor:
        # assert input.dim() == 4 and input.size(2)*input.size(3)==self.channels
        return self.scale*(input+self.shift)



class StraightThrough(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

