import torch

from model.res_unet import ResUNet


if __name__ == "__main__":
    images = torch.rand((4, 3, 256, 256))
    model = ResUNet(10)
    res = model(images)
    print(res.shape)