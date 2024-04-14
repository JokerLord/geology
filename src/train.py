import numpy as np
import argparse

import torch
import torch.nn.functional as F

from res_unet import ResUNet
from dataset import LumenStoneDataset
from model import Trainer
from config import *
from utils import present_class_codes


def train(gpu_index: int):
    model = ResUNet(len(present_class_codes), 3, N_FILTERS, True)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.1,
        patience=4
    )
    trainer = Trainer(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device(f"cuda:{gpu_index}"),
        max_epochs=50,
    )

    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu_index", type=int, help="GPU index")

    args = parser.parse_args()
    train(args.gpu_index)
