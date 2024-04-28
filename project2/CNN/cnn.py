# We based our CNN on this: https://medium.com/@chen-yu/building-a-customized-residual-cnn-with-pytorch-471810e894ed

import torch
import torch.nn as nn
from torch import Tensor

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

        #Out features from the 4 blocks of residual layers: 128
        
        
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