import torch
from torch import nn

class resnet_block(nn.Module):
    def __init__(self ,in_channels,channels,stride=1,downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.conv3 = nn.Conv2d(in_channels=channels, out_channels=channels * 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
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
class ResNet50(nn.Module):
    def __init__(self,num_classes=1000):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 3)
        self.layer2 = self._make_layer(256, 128, 4, stride=2)
        self.layer3 = self._make_layer(512, 256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channels, channels, blocks, stride=1):
        downsample = None
        if stride != 1 or in_channels != channels * 4:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=channels * 4, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels * 4)
            )

        layers = []
        layers.append(resnet_block(in_channels, channels, stride, downsample))
        for _ in range(1, blocks):
            layers.append(resnet_block(channels * 4, channels))

        return nn.Sequential(*layers)