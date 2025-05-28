import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.relu(self.bn(self.conv1(x)))
        out = self.bn(self.conv2(out))

        out += residual
        out = self.relu(out)

        return out

class FeatureExtractor(nn.Module):
    def __init__(self, num_channels, num_blocks=4):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()

        blocks = []
        for _ in range(num_blocks):
            blocks.append(ResidualBlock(num_channels, num_channels))

        self.residual_blocks = nn.Sequential(*blocks)

    def forward(self, x):
        out = self.relu(self.conv(x))
        out = self.residual_blocks(out)

        return out
