"""ResNet50（3/4/6/3 Bottleneck），可选在残差相加前加入 SE 通道注意力。"""

from pathlib import Path

import torch
from torch import nn
from torchvision.models import ResNet50_Weights


OFFICIAL_WEIGHTS = ResNet50_Weights.IMAGENET1K_V2


class SEBlock(nn.Module):
    """全局平均池化 → 通道压缩 → ReLU → 通道恢复 → Sigmoid。"""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        if channels < 1 or reduction < 1:
            raise ValueError("channels 和 reduction 必须大于0")
        hidden = max(1, channels // reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 始终保留 N,C,1,1 四维形状，供 1×1 卷积与广播相乘使用。
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(self.avgpool(x)))))
        return x * scale


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None,
                 use_se=False, se_reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        # 与 torchvision ResNet50 V1.5 一致：下采样步长放在 3×3 卷积。
        self.conv2 = nn.Conv2d(channels, channels, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(channels, channels * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.se = SEBlock(channels * self.expansion, se_reduction) if use_se else nn.Identity()

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.se(self.bn3(self.conv3(out)))
        return self.relu(out + identity)


class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, use_se=False, se_reduction=16, init_weights=True):
        super().__init__()
        if num_classes < 1 or se_reduction < 1:
            raise ValueError("num_classes 和 se_reduction 必须大于0")
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 64, 3, 1, use_se, se_reduction)
        self.layer2 = self._make_layer(256, 128, 4, 2, use_se, se_reduction)
        self.layer3 = self._make_layer(512, 256, 6, 2, use_se, se_reduction)
        self.layer4 = self._make_layer(1024, 512, 3, 2, use_se, se_reduction)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(2048, num_classes)
        if init_weights:
            for module in self.modules():
                if isinstance(module, nn.Conv2d):
                    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.BatchNorm2d):
                    nn.init.ones_(module.weight)
                    nn.init.zeros_(module.bias)

    @staticmethod
    def _make_layer(in_channels, channels, blocks, stride, use_se, se_reduction):
        out_channels = channels * Bottleneck.expansion
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        layers = [Bottleneck(in_channels, channels, stride, downsample, use_se, se_reduction)]
        layers.extend(Bottleneck(out_channels, channels, use_se=use_se, se_reduction=se_reduction)
                      for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(torch.flatten(self.avgpool(x), 1))


def build_resnet50(num_classes=1000, use_se=False, se_reduction=16,
                   pretrained=False, weights_path: str | Path | None = None,
                   init_weights=True) -> ResNet50:
    """加载标准 ResNet50 的官方1000类权重；SE 参数另行初始化。

    weights_path 接受 torchvision 原始 ResNet50 state_dict。
    完整训练检查点（含 SE 参数）由训练脚本的 test 命令恢复。
    """
    if weights_path is not None and not pretrained:
        raise ValueError("使用官方权重文件时，请设置 pretrained=True")
    if num_classes < 1 or se_reduction < 1:
        raise ValueError("num_classes 和 se_reduction 必须大于0")
    model = ResNet50(
        num_classes=1000 if pretrained else num_classes,
        use_se=use_se and not pretrained,
        se_reduction=se_reduction,
        init_weights=init_weights and not pretrained,
    )
    if pretrained:
        state_dict = (
            OFFICIAL_WEIGHTS.get_state_dict(progress=True, check_hash=True)
            if weights_path is None
            else torch.load(weights_path, map_location="cpu", weights_only=True)
        )
        # 先严格检查标准骨干的所有参数，避免 strict=False 静默漏载。
        model.load_state_dict(state_dict, strict=True)
        if use_se:
            for module in model.modules():
                if isinstance(module, Bottleneck):
                    module.se = SEBlock(module.bn3.num_features, se_reduction)
        if num_classes != 1000:
            model.fc = nn.Linear(2048, num_classes)
    return model
