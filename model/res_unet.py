import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union


class ConvRes(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        BN: bool = True,
        kernel_size: Union[int, tuple] = 3,
        padding: Union[int, tuple, str] = "same",
    ):
        super().__init__()

        """ Shortcut Connection """
        self.shortcut = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=padding
        )
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

        """ Convolution layer """
        self.BN = BN
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)

        """ Initialization """
        nn.init.xavier_normal_(self.conv1.weight)
        nn.init.xavier_normal_(self.conv2.weight)

    def forward(self, inputs):
        s = self.shortcut_bn(self.shortcut(inputs))

        x = F.relu(self.conv1(inputs))
        if self.BN:
            x = self.bn1(x)
        x = F.relu(self.conv2(x))
        if self.BN:
            x = self.bn2(x)

        return x + s


class EncoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, BN: bool = True):
        super().__init__()

        self.conv_res = ConvRes(in_channels, out_channels, BN)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv_res(inputs)
        p = self.pool(x)
        return x, p


# class DecoderBlock(nn.Module):
#     def __init__(self, in_channels: int, out_channels: int, BN: bool = True):
#         super().__init__()

#         self.upsample = nn.ConvTranspose2d(
#             in_channels, out_channels, kernel_size=2, stride=2, padding="same"
#         )
#         self.conv = ConvRes(in_channels, out_channels, BN)

#     def forward(self, inputs, shortcut):
#         x = self.upsample(inputs)
#         print(x.shape, shortcut.shape)


class ResUNet(nn.Module):
    def __init__(
        self, n_classes: int, n_channels: int = 3, filters: int = 16, BN: bool = True
    ):
        super().__init__()

        """ Encoder """
        self.en1 = EncoderBlock(n_channels, filters, BN)

    def forward(self, inputs):
        x, p = self.en1(inputs)


if __name__ == "__main__":
    inputs = torch.randn([4, 3, 256, 256])
    model = ResUNet(10)
    model(inputs)
