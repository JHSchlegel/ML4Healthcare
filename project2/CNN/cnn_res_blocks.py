# We based our CNN on this: https://medium.com/@chen-yu/building-a-customized-residual-cnn-with-pytorch-471810e894ed


import torch
import torch.nn as nn
from torch import Tensor

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding='same', 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=(3, 3), 
            padding='same', 
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels, 
                    out_channels, 
                    kernel_size=(3, 3), 
                    padding='same', 
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = self.downsample(x) if self.downsample else x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x + identity)

        return x
    
    
class ResidualCnn(nn.Module):
    def __init__(self, classes_num: int, in_channels: int = 1, image_size: tuple = (128, 128)):
        super().__init__()

        # Initial convolution layer
        self.conv = nn.Conv2d(
            in_channels=in_channels, 
            out_channels=16, 
            kernel_size=(3, 3), 
            padding='same', 
            bias=False
        )
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU()

        # First block of residual layers and pooling
        self.layer1 = nn.Sequential(
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            ResidualBlock(16, 16),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # Second block of residual layers and pooling
        self.layer2 = nn.Sequential(
            ResidualBlock(16, 32),
            ResidualBlock(32, 32),
            ResidualBlock(32, 32),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # Third block of residual layers and pooling
        self.layer3 = nn.Sequential(
            ResidualBlock(32, 64),
            ResidualBlock(64, 64),
            ResidualBlock(64, 64),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # Fourth block of residual layers and pooling
        self.layer4 = nn.Sequential(
            ResidualBlock(64, 128),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        
        # Flattening and final linear layer
        self.flatten = nn.Flatten(1)
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(
            in_features=int(image_size[0] / 2**4 * image_size[1] / 2**4 * 128),
            out_features=classes_num
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)

        return x