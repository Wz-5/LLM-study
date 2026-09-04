import torch
from torch import nn

class DepthwiseConv2d(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1, padding=0, bias=False):
        super(DepthwiseConv2d, self).__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=bias,
        )

    def forward(self, x):
        return self.depthwise(x)

class PointwiseConv2d(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size=1,stride=1,padding=0,bias=False):
        super().__init__()
        self.pointwise = nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size,stride=stride,padding=padding,bias=bias,groups=1)

    def forward(self,x):
        return self.pointwise(x)