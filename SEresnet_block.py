import torch
from torch import nn

class SEresnet_block(nn.Module):
    def __init__(self,in_channels,channels):
        super().__init__()

        self.conv1=nn.Conv2d(in_channels,channels,kernel_size=1,stride=1,padding=0)
        self.bn1=nn.BatchNorm2d(channels)
        self.relu=nn.ReLU(inplace=True)

        self.conv2=nn.Conv2d(channels,channels,kernel_size=3,stride=1,padding=1)
        self.bn2=nn.BatchNorm2d(channels)
        self.relu2=nn.ReLU(inplace=True)

        self.conv3=nn.Conv2d(channels,channels*4,kernel_size=1,stride=1,padding=0)
        self.bn3=nn.BatchNorm2d(channels*4)

        self.se=SEBlock(channels*4)

    def forward(self,x):
        identity=x

        out=self.conv1(x)
        out=self.bn1(out)
        out=self.relu(out)

        out=self.conv2(out)
        out=self.bn2(out)
        out=self.relu2(out)

        out=self.conv3(out)
        out=self.bn3(out)

        out=self.se(out)

        out+=identity
        out=self.relu(out)

        return out

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(in_channels, in_channels // reduction,kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels // reduction, in_channels,kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        y = self.global_avg_pool(x).view(batch_size, channels)
        y = self.conv1(y)
        y = self.relu(y)
        y = self.conv2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        return x * y