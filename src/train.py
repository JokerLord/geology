import numpy as np
import argparse

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from res_unet import ResUNet
from dataset import LumenStoneDataset
from datamodule import LumenStoneDataModule
from model import Trainer
from config import *


def train(gpu_index: int):
    model = ResUNet(len(CLASS_NAMES), 3, N_FILTERS, True)

    trainer = Trainer(
        model=model,
        criterion=torch.nn.CrossEntropyLoss(),
        optimizer=torch.optim.Adam(model.parameters(), lr=LR),
        device=torch.device(f"cuda:{gpu_index}"),
        max_epochs=1,
    )

    trainer.fit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("gpu_index", type=int, help="GPU index")

    args = parser.parse_args()
    train(args.gpu_index)
