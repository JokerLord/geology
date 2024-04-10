import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset, DataLoader
from torch import Tensor

from res_unet import ResUNet
from dataset import LumenStoneDataset
from utils import split_into_patches, combine_from_patches
from config import *

import random
from sklearn.model_selection import train_test_split
from typing import Callable


class Trainer:
    def __init__(self, model, criterion, optimizer, device, max_epochs) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.max_epochs = max_epochs

        self._train_dataset = LumenStoneDataset(
            root_dir=LUMENSTONE_PATH, train=True, transform=TRAIN_TRANSFORM
        )
        self._val_dataset = LumenStoneDataset(
            root_dir=LUMENSTONE_PATH, train=True, transform=VAL_TRANSFORM
        )

    def _split_dataset(self) -> tuple[DataLoader, DataLoader]:
        indices = list(range(len(self._train_dataset)))
        train_indices, val_indices = train_test_split(
            indices, train_size=SPLIT_RATIO, shuffle=True)
        train_dataset = Subset(self._train_dataset, train_indices)
        val_dataset = Subset(self._val_dataset, val_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4)
        return train_dataloader, val_dataloader

    def _train_one_image(self, inputs: Tensor, target: Tensor) -> float:
        """
        Arguments:
            inputs: Input image of size (3, H, W)
            target: Image mask of size (H, W)
        Returns:
            image_loss (float): mean loss for the whole image
        """

        image_loss = 0
        input_patches = split_into_patches(
            inputs,
            patch_size=PATCH_SIZE,
            offset=OFFSET,
            overlap=PATCH_OVERLAP
        )
        target_patches = split_into_patches(
            target,
            patch_size=PATCH_SIZE,
            offset=OFFSET,
            overlap=PATCH_OVERLAP
        )

        """ Make total number of patches a multiple of BATCH_SIZE """
        while (len(input_patches) % BATCH_SIZE != 0):
            input_patches.append(input_patches[-1])
            target_patches.append(target_patches[-1])

        """ Shuffle patches """
        zipped_patches = list(zip(input_patches, target_patches))
        random.shuffle(zipped_patches)
        input_patches, target_patches = zip(*zipped_patches)

        for i in range(0, len(input_patches), BATCH_SIZE):
            batch = torch.stack(input_patches[i: i + BATCH_SIZE]).to(device=self.device, dtype=torch.float32)
            logits_batch = self.model(batch)
            target_batch = torch.stack(target_patches[i: i + BATCH_SIZE]).to(device=self.device, dtype=torch.int64)
            loss = self.criterion(logits_batch, target_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            image_loss += loss.item()

        return image_loss / (len(input_patches) // BATCH_SIZE)

    def fit(self):
        self.model.to(self.device)
        for epoch in range(self.max_epochs):
            train_dataloader, val_dataloader = self._split_dataset()

            self.model.train()
            train_loss = 0
            for batch in train_dataloader:
                inputs, target = batch
                image_loss = self._train_one_image(inputs[0], target[0])
                train_loss += image_loss
            train_loss = train_loss / len(train_dataloader)

            with torch.no_grad():
                for batch in val_dataloader:
                    inputs, target = batch
            print(f"Epoch {epoch}: train_loss: {train_loss:.3f}")

                    # logits_patches = logits_patches[:init_n_patches]

                    # logits = combine_from_patches(
                    #     logits_patches,
                    #     patch_size=PATCH_SIZE,
                    #     offset=OFFSET,
                    #     overlap=PATCH_OVERLAP,
                    #     src_shape=inputs.shape[-2:],
                    # )

                    # logits = logits[None, ...]
                    # target = target[None, ...]

                    # activated = F.softmax(input=logits, dim=1)
                    # predictions = torch.argmax(activated, dim=1)

                    # loss = self.criterion(logits, target.to(torch.int64))
                    # self.optimizer.zero_grad()
                    # loss.backward()
                    # self.optimizer.step()
