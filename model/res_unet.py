import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Union


class ConvRes(nn.Module):
    def __init__(
        self, in_channels, out_channels, BN=True, kernel_size=3, padding="same"
    ):
        """
        Arguments:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the convolution
            BN (bool, optional): If True, enables batch normalization. Default: True.
            kernel_size (int or tuple, optional): Size of the convolving kernel. Default: 3.
            padding (int, tuple or str, optional): Padding added to all four sides of the input. Default: "same".
        """

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
    def __init__(self, in_channels, out_channels, BN=True):
        """
        Arguments:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the encoding block.
            BN (bool, optional): If True, enables batch normalization. Default: True.
        """

        super().__init__()

        self.conv_res = ConvRes(in_channels, out_channels, BN)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv_res(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, BN=True):
        """
        Arguments:
            in_channels (int): Number of channels in the input tensor.
            out_channels (int): Number of channels produced by the decoding block.
            BN (bool, optional): If True, enables batch normalization. Default: True.
        """

        super().__init__()

        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2, padding=0
        )
        self.conv = ConvRes(out_channels * 2, out_channels, BN)

    def forward(self, inputs, shortcut):
        x = self.upsample(inputs)
        x = torch.cat([x, shortcut], axis=1)
        x = self.conv(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, n_classes, n_channels=3, filters=16, BN=True):
        """
        Arguments:
            n_classes (int): Number of output classes.
            n_channels (int, optional): Number of channels in input images. Default: 3.
            filters (int, optional): Number of filters in the first ConvRes block. Default: 16.
            BN (bool, optional): If True, enables batch normalization. Default: True.
        """

        super().__init__()

        """ Encoder """
        self.encoder1 = EncoderBlock(
            n_channels, filters, BN
        )  # (N, 16, x.H / 2, x.W / 2)
        self.encoder2 = EncoderBlock(
            filters, filters * 2, BN
        )  # (N, 32, x.H / 4, x.W / 4)
        self.encoder3 = EncoderBlock(
            filters * 2, filters * 4, BN
        )  # (N, 64, x.H / 8, x.W / 8)
        self.encoder4 = EncoderBlock(
            filters * 4, filters * 8, BN
        )  # (N, 128, x.H / 16, x.W / 16)

        """ Bridge """
        self.bridge = ConvRes(
            filters * 8, filters * 16, BN
        )  # (N, 256, x.H / 16, x.W / 16)

        """ Decoder """
        self.decoder1 = DecoderBlock(
            filters * 16, filters * 8, BN
        )  # (N, 128, x.H / 8, x.W / 8)
        self.decoder2 = DecoderBlock(
            filters * 8, filters * 4, BN
        )  # (N, 64, x.H / 4, x.W / 4)
        self.decoder3 = DecoderBlock(
            filters * 4, filters * 2, BN
        )  # (N, 32, x.H / 2, x.W / 2)
        self.decoder4 = DecoderBlock(filters * 2, filters, BN)  # (N, 16, x.H, x.W)

        """ Classifier """
        self.outputs = nn.Conv2d(filters, n_classes, kernel_size=1)  # (N, 16, x.H, x.W)

    def forward(self, inputs):
        s1, p1 = self.encoder1(inputs)
        s2, p2 = self.encoder2(p1)
        s3, p3 = self.encoder3(p2)
        s4, p4 = self.encoder4(p3)

        x = self.bridge(p4)

        x = self.decoder1(x, s4)
        x = self.decoder2(x, s3)
        x = self.decoder3(x, s2)
        x = self.decoder4(x, s1)

        x = F.softmax(self.outputs(x), dim=1)

        return x
