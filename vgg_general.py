"""可切换 VGG-11/13/16/19 的通用实现。

示例：
    # ImageNet 预训练 VGG-11，修改成猫狗二分类
    model = build_vgg(depth=11, num_classes=2, pretrained=True)

    # ImageNet 预训练 VGG-16-BN，修改成十分类
    model = build_vgg(
        depth=16,
        num_classes=10,
        batch_norm=True,
        pretrained=True,
    )

    # 不加载预训练权重的 VGG-19
    model = build_vgg(depth=19, num_classes=10, pretrained=False)
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple, Union

import torch
from torch import nn
from torchvision.models import (
    VGG11_BN_Weights,
    VGG11_Weights,
    VGG13_BN_Weights,
    VGG13_Weights,
    VGG16_BN_Weights,
    VGG16_Weights,
    VGG19_BN_Weights,
    VGG19_Weights,
)


LayerConfig = Union[int, str]


# A、B、D、E 分别对应 VGG-11、VGG-13、VGG-16、VGG-19。
VGG_CONFIGS: Dict[int, List[LayerConfig]] = {
    11: [
        64, "M",
        128, "M",
        256, 256, "M",
        512, 512, "M",
        512, 512, "M",
    ],
    13: [
        64, 64, "M",
        128, 128, "M",
        256, 256, "M",
        512, 512, "M",
        512, 512, "M",
    ],
    16: [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, "M",
        512, 512, 512, "M",
        512, 512, 512, "M",
    ],
    19: [
        64, 64, "M",
        128, 128, "M",
        256, 256, 256, 256, "M",
        512, 512, 512, 512, "M",
        512, 512, 512, 512, "M",
    ],
}


# depth 和 batch_norm 共同决定加载哪一种官方权重。
OFFICIAL_WEIGHTS = {
    (11, False): VGG11_Weights.DEFAULT,
    (11, True): VGG11_BN_Weights.DEFAULT,
    (13, False): VGG13_Weights.DEFAULT,
    (13, True): VGG13_BN_Weights.DEFAULT,
    (16, False): VGG16_Weights.DEFAULT,
    (16, True): VGG16_BN_Weights.DEFAULT,
    (19, False): VGG19_Weights.DEFAULT,
    (19, True): VGG19_BN_Weights.DEFAULT,
}


def make_features(
    config: Sequence[LayerConfig],
    batch_norm: bool = False,
    in_channels: int = 3,
) -> nn.Sequential:
    """根据配置表构造 VGG 的卷积特征提取部分。

    所有层直接放在同一个 Sequential 中，使参数名称与 torchvision
    官方模型保持一致，例如 features.0.weight。
    """
    layers: List[nn.Module] = []

    for value in config:
        if value == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue

        out_channels = int(value)
        convolution = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
        )

        if batch_norm:
            layers.extend(
                [
                    convolution,
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
        else:
            layers.extend(
                [
                    convolution,
                    nn.ReLU(inplace=True),
                ]
            )

        in_channels = out_channels

    return nn.Sequential(*layers)


class VGG(nn.Module):
    """通用 VGG 网络。

    属性名称 features、avgpool、classifier 与 torchvision 官方实现一致，
    因此只要卷积配置相同，就可以严格加载相应的官方 state_dict。
    """

    def __init__(
        self,
        features: nn.Module,
        num_classes: int = 1000,
        dropout: float = 0.5,
        init_weights: bool = True,
    ) -> None:
        super().__init__()

        self.features = features
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes),
        )

        if init_weights:
            self.initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)

    def initialize_weights(self) -> None:
        """使用 torchvision VGG 相同的参数初始化方式。"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
                nn.init.zeros_(module.bias)


def build_vgg(
    depth: int = 11,
    num_classes: int = 1000,
    batch_norm: bool = False,
    pretrained: bool = False,
    freeze_features: bool = False,
    in_channels: int = 3,
    dropout: float = 0.5,
) -> VGG:
    """通过少量参数构建不同版本的 VGG。

    Args:
        depth: VGG 深度，只能是 11、13、16、19。
        num_classes: 下游任务类别数，例如猫狗分类为 2。
        batch_norm: 是否在每个卷积层后使用 BatchNorm。
        pretrained: 是否加载对应的 torchvision ImageNet 官方权重。
        freeze_features: 是否冻结卷积特征提取层。
        in_channels: 输入通道数；加载官方权重时必须为 3。
        dropout: 分类器的 Dropout 概率。
    """
    if depth not in VGG_CONFIGS:
        raise ValueError(
            f"不支持 VGG-{depth}，可选深度为 {tuple(VGG_CONFIGS)}"
        )

    if pretrained and in_channels != 3:
        raise ValueError(
            "官方 ImageNet 预训练权重要求 in_channels=3；"
            "灰度图片请在数据预处理阶段转换成 RGB"
        )

    # 官方预训练权重的分类器输出为 ImageNet 1000 类。
    initial_num_classes = 1000 if pretrained else num_classes
    model = VGG(
        features=make_features(
            VGG_CONFIGS[depth],
            batch_norm=batch_norm,
            in_channels=in_channels,
        ),
        num_classes=initial_num_classes,
        dropout=dropout,
        init_weights=not pretrained,
    )

    if pretrained:
        weights = OFFICIAL_WEIGHTS[(depth, batch_norm)]
        state_dict = weights.get_state_dict(
            progress=True,
            check_hash=True,
        )

        # strict=True：参数名称和张量尺寸必须全部匹配。
        model.load_state_dict(state_dict, strict=True)

        # 先加载1000类权重，再替换为当前任务的分类层。
        if num_classes != 1000:
            model.classifier[6] = nn.Linear(4096, num_classes)
            nn.init.normal_(model.classifier[6].weight, mean=0.0, std=0.01)
            nn.init.zeros_(model.classifier[6].bias)

    if freeze_features:
        for parameter in model.features.parameters():
            parameter.requires_grad = False

    return model


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """返回总参数量和可训练参数量。"""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return total, trainable


if __name__ == "__main__":
    # 日常使用时，通常只需要修改下面这些参数。
    DEPTH = 11
    NUM_CLASSES = 2
    BATCH_NORM = False
    PRETRAINED = False
    FREEZE_FEATURES = False

    network = build_vgg(
        depth=DEPTH,
        num_classes=NUM_CLASSES,
        batch_norm=BATCH_NORM,
        pretrained=PRETRAINED,
        freeze_features=FREEZE_FEATURES,
    )

    total_parameters, trainable_parameters = count_parameters(network)
    print(network)
    print(f"总参数量：{total_parameters:,}")
    print(f"可训练参数量：{trainable_parameters:,}")

    # 用较小的 batch 验证输出尺寸。正式训练通常使用 224×224 输入。
    sample = torch.randn(1, 3, 224, 224)
    output = network(sample)
    print("输出尺寸：", output.shape)
