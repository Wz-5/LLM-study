import torch
from torch import nn

class inception(nn.Module):
    def __init__(
        self,
        in_channels,
        cn1,
        cn3_reduce,
        cn3,
        cn5_reduce,
        cn_5,
        cn_pool=None,
    ):
        super().__init__()
        if cn_pool is None:
            cn_pool = in_channels

        self.branch1=nn.Sequential(
            nn.Conv2d(
                in_channels,
                cn1,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True)
        )
        self.branch2=nn.Sequential(
            nn.Conv2d(
                in_channels,
                cn3_reduce,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                cn3_reduce,
                cn3,
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.ReLU(inplace=True)
        )
        self.branch3=nn.Sequential(
            nn.Conv2d(
                in_channels,
                cn5_reduce,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                cn5_reduce,
                cn_5,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(inplace=True)
        )
        self.branch4=nn.Sequential(
            nn.MaxPool2d(
                kernel_size=3,
                stride=1,
                padding=1
            ),
            nn.Conv2d(
                in_channels,
                cn_pool,
                kernel_size=1,
                stride=1,
                padding=0
            ),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x1=self.branch1(x)
        x2=self.branch2(x)
        x3=self.branch3(x)
        x4=self.branch4(x)

        return torch.cat([x1,x2,x3,x4], dim=1)
