import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, n_classes: int, n_channels: int, n_filters: int, BN: bool):
        """
        Arguments:
            n_classes (int): Number of output classes.
            n_channels (int): Number of channels in input images.
            n_filters (int): Number of n_filters in the first ConvRes block. Default: 16.
            BN (bool): If True, enables batch normalization.
        """
        super().__init__()

        # Encoder
        # input: (3, 384, 384)
        self.e11 = nn.Conv2d(
            n_channels, n_filters, kernel_size=3, padding=1
        )  # output: (16, 384, 384)
        self.e12 = nn.Conv2d(
            n_filters, n_filters, kernel_size=3, padding=1
        )  # output: (16, 384, 384)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: (16, 192, 192)

        # input: (16, 192, 192)
        self.e21 = nn.Conv2d(
            n_filters, n_filters * 2, kernel_size=3, padding=1
        )  # output: (32, 192, 192)
        self.e22 = nn.Conv2d(
            n_filters * 2, n_filters * 2, kernel_size=3, padding=1
        )  # output: (32, 192, 192)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: (32, 96, 96)

        # input: (32, 96, 96)
        self.e31 = nn.Conv2d(
            n_filters * 2, n_filters * 4, kernel_size=3, padding=1
        )  # output: (64, 96, 96)
        self.e32 = nn.Conv2d(
            n_filters * 4, n_filters * 4, kernel_size=3, padding=1
        )  # output: (64, 96, 96)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: (64, 48, 48)

        # input: (64, 48, 48)
        self.e41 = nn.Conv2d(
            n_filters * 4, n_filters * 8, kernel_size=3, padding=1
        )  # output: (128, 48, 48)
        self.e42 = nn.Conv2d(
            n_filters * 8, n_filters * 8, kernel_size=3, padding=1
        )  # output: (128, 48, 48)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)  # output: (128, 24, 24)

        # input: (128, 24, 24)
        self.e51 = nn.Conv2d(
            n_filters * 8, n_filters * 16, kernel_size=3, padding=1
        )  # output: (256, 24, 24)
        self.e52 = nn.Conv2d(
            n_filters * 16, n_filters * 16, kernel_size=3, padding=1
        )  # output: (256, 24, 24)

        # Decoder
        self.upconv1 = nn.ConvTranspose2d(
            n_filters * 16, n_filters * 8, kernel_size=2, stride=2
        )
        self.d11 = nn.Conv2d(n_filters * 16, n_filters * 8, kernel_size=3, padding=1)
        self.d12 = nn.Conv2d(n_filters * 8, n_filters * 8, kernel_size=3, padding=1)

        self.upconv2 = nn.ConvTranspose2d(
            n_filters * 8, n_filters * 4, kernel_size=2, stride=2
        )
        self.d21 = nn.Conv2d(n_filters * 8, n_filters * 4, kernel_size=3, padding=1)
        self.d22 = nn.Conv2d(n_filters * 4, n_filters * 4, kernel_size=3, padding=1)

        self.upconv3 = nn.ConvTranspose2d(
            n_filters * 4, n_filters * 2, kernel_size=2, stride=2
        )
        self.d31 = nn.Conv2d(n_filters * 4, n_filters * 2, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(n_filters * 2, n_filters * 2, kernel_size=3, padding=1)

        self.upconv4 = nn.ConvTranspose2d(
            n_filters * 2, n_filters, kernel_size=2, stride=2
        )
        self.d41 = nn.Conv2d(n_filters * 2, n_filters, kernel_size=3, padding=1)
        self.d42 = nn.Conv2d(n_filters, n_filters, kernel_size=3, padding=1)

        self.outconv = nn.Conv2d(n_filters, n_classes, kernel_size=1)

    def forward(self, x):
        """ Encoder """
        xe11 = F.relu(self.e11(x))
        xe12 = F.relu(self.e12(xe11))
        xp1 = self.pool1(xe12)

        xe21 = F.relu(self.e21(xp1))
        xe22 = F.relu(self.e22(xe21))
        xp2 = self.pool2(xe22)

        xe31 = F.relu(self.e31(xp2))
        xe32 = F.relu(self.e32(xe31))
        xp3 = self.pool3(xe32)

        xe41 = F.relu(self.e41(xp3))
        xe42 = F.relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = F.relu(self.e51(xp4))
        xe52 = F.relu(self.e52(xe51))

        """ Decoder """
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = F.relu(self.d11(xu11))
        xd12 = F.relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = F.relu(self.d21(xu22))
        xd22 = F.relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = F.relu(self.d31(xu33))
        xd32 = F.relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = F.relu(self.d41(xu44))
        xd42 = F.relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
