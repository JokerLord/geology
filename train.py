import numpy as np

import torch
import torch.nn.functional as F

import pytorch_lightning as pl

from model.res_unet import ResUNet
from dataset import LumenStoneDataset
from datamodule import LumenStoneDataModule
from model.model import LumenStoneSegmentation
from config import *
from utils.patches import split_into_patches, combine_from_patches


if __name__ == "__main__":
    trainer = pl.Trainer(max_epochs=1)
    datamodule = LumenStoneDataModule(
        root_dir=LUMENSTONE_PATH,
        batch_size=BATCH_SIZE,
        split_ratio=SPLIT_RATIO,
        train_transform=TRAIN_TRANSFORM,
        val_transform=VAL_TRANSFORM,
    )

    model = LumenStoneSegmentation(
        n_classes=len(CLASS_NAMES),
        n_channels=3,
        n_filters=N_FILTERS,
        BN=True,
        loss_func=torch.nn.CrossEntropyLoss,
        optimizer=torch.optim.Adam,
        lr=LR,
    )

    trainer.fit(model=model, datamodule=datamodule)
