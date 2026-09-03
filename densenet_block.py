import torch
from torch import nn

class Denselayer(nn.Module):
    def __init__(self,in_channels,bn_size=4,grow_rate=32,drop_rate=0):
        super().__init__()
        bottleneck_channels=bn_size*grow_rate

        self.norm1=nn.BatchNorm2d(in_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv1=nn.Conv2d(in_channels,bottleneck_channels,kernel_size=1,stride=1,padding=0)
        self.norm2=nn.BatchNorm2d(bottleneck_channels)
        self.conv2=nn.Conv2d(bottleneck_channels,grow_rate,kernel_size=3,stride=1,padding=1)
        self.drop=(nn.Dropout2d(p=drop_rate)
                    if drop_rate>0 else nn.Identity())
        def forward(self,x):
            out=self.conv1(self.relu(self.norm1(x)))
            out=self.conv2(self.relu(self.norm2(out)))
            out=self.drop(out)
            out=torch.cat([x,out],dim=1)
            return out

class DenseBlock(nn.Module):
    def __init__(self,num_layers,in_channels,bn_size=4,grow_rate=32,drop_rate=0):
        super().__init__()
        self.layers=nn.ModuleList()
        for i in range(num_layers):
            layer=Denselayer(in_channels+grow_rate*i,bn_size,grow_rate,drop_rate)
            self.layers.append(layer)

    def forward(self,x):
        for layer in self.layers:
            x=layer(x)
        return x

class Transition(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.norm=nn.BatchNorm2d(in_channels)
        self.relu=nn.ReLU(inplace=True)
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,padding=0)
        self.pool=nn.AvgPool2d(kernel_size=2,stride=2)

    def forward(self,x):
        out=self.conv(self.relu(self.norm(x)))
        out=self.pool(out)
        return out
        
