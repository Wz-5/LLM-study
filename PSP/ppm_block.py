import torch
from torch import nn
from torch.nn import functional as F


class PPMBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_sizes=(1, 2, 3, 6)):
        super().__init__()
        pool_sizes = tuple(pool_sizes)
        branch_channels = in_channels // len(pool_sizes)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(size),
                nn.Conv2d(in_channels, branch_channels, 1, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True),
            )
            for size in pool_sizes
        ])

        concat_channels = in_channels + len(pool_sizes) * branch_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(concat_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        spatial_size = x.shape[-2:]
        features = [x]
        for branch in self.branches:
            pooled = branch(x)
            upsampled = F.interpolate(
                pooled, size=spatial_size, mode="bilinear", align_corners=False
            )
            features.append(upsampled)

        return self.fuse(torch.cat(features, dim=1))


