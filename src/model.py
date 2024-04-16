import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Subset, DataLoader
from torch import Tensor

from res_unet import ResUNet
from dataset import LumenStoneDataset
from utils import *
from config import *
from metrics import iou_per_class, accuracy
from evaluation import EvaluationResult, evaluate_dataset

import random
import copy
from sklearn.model_selection import train_test_split
from typing import Callable
from tqdm.auto import tqdm
from collections import defaultdict


class Trainer:
    def __init__(
        self, model, criterion, optimizer, scheduler, device, max_epochs, exp_path
    ) -> None:
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_epochs = max_epochs
        self.exp_path = exp_path

        self._train_dataset = LumenStoneDataset(
            root_dir=LUMENSTONE_PATH, train=True, transform=TRAIN_TRANSFORM
        )
        self._val_dataset = LumenStoneDataset(
            root_dir=LUMENSTONE_PATH, train=True, transform=VAL_TRANSFORM
        )
        self._test_dataset = LumenStoneDataset(
            root_dir=LUMENSTONE_PATH, train=False, transform=VAL_TRANSFORM
        )

        self.best_model = None
        self.best_avg_val_loss = 5.0 # TODO

        self.log = open(self.exp_path / "metrics.txt", "a+")
        self.log_detailed = open(self.exp_path / "metrics_detailed.txt", "a+")


    def _split_dataset(self) -> tuple[DataLoader, DataLoader]:
        indices = list(range(len(self._train_dataset)))
        train_indices, val_indices = train_test_split(
            indices, train_size=SPLIT_RATIO, shuffle=True
        )
        train_dataset = Subset(self._train_dataset, train_indices)
        val_dataset = Subset(self._val_dataset, val_indices)
        train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=1, num_workers=4)
        return train_dataloader, val_dataloader


    def _train_image(self, inputs: Tensor, target: Tensor) -> float:
        inputs, target = torch.squeeze(inputs), torch.squeeze(target) # (3, H, W), (H, W)

        inputs_patches = split_into_patches(
            image=inputs, patch_size=PATCH_SIZE, offset=OFFSET, overlap=PATCH_OVERLAP
        )
        target_patches = split_into_patches(
            image=target, patch_size=PATCH_SIZE, offset=OFFSET, overlap=PATCH_OVERLAP
        )

        """ Make total number of patches a multiple of BATCH_SIZE """
        while len(inputs_patches) % BATCH_SIZE != 0:
            inputs_patches.append(inputs_patches[-1])
            target_patches.append(target_patches[-1])

        """ Shuffle patches """
        zipped_patches = list(zip(inputs_patches, target_patches))
        random.shuffle(zipped_patches)
        inputs_patches, target_patches = zip(*zipped_patches)

        losses = []
        for i in range(0, len(inputs_patches), BATCH_SIZE):
            inputs_batch = torch.stack(inputs_patches[i : i + BATCH_SIZE]).to(
                device=self.device
            ) # (batch_size, 3, patch_size, patch_size)
            logits_batch = self.model(inputs_batch)  # (batch_size, n_classes, patch_size, patch_size)
            target_batch = torch.stack(target_patches[i : i + BATCH_SIZE]).to(
                device=self.device
            )  # (batch_size, patch_size, patch_size)
            loss = self.criterion(logits_batch, target_batch)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        train_loss = sum(losses) / len(losses)
        return train_loss

    
    def _validate_image(self, inputs: Tensor, target: Tensor) -> tuple[float, EvaluationResult]:
        inputs, target = torch.squeeze(inputs), torch.squeeze(target) # (3, H, W), (n_classes, H, W)

        inputs_patches = split_into_patches(
            image=inputs, patch_size=PATCH_SIZE, offset=OFFSET, overlap=PATCH_OVERLAP
        )
        target_patches = split_into_patches(
            image=target, patch_size=PATCH_SIZE, offset=OFFSET, overlap=PATCH_OVERLAP
        )

        """ Make total number of patches a multiple of BATCH_SIZE """
        init_n_patches = len(inputs_patches)
        while len(inputs_patches) % BATCH_SIZE != 0:
            inputs_patches.append(inputs_patches[-1])
            target_patches.append(target_patches[-1])

        with torch.no_grad():
            logits_patches = []
            losses = []
            for i in range(0, len(inputs_patches), BATCH_SIZE):
                inputs_batch = torch.stack(inputs_patches[i : i + BATCH_SIZE]).to(
                    device=self.device
                ) # (batch_size, 3, patch_size, patch_size)
                logits_batch = self.model(inputs_batch)  # (batch_size, n_classes, patch_size, patch_size)
                target_batch = torch.stack(target_patches[i: i + BATCH_SIZE]).to(
                    device=self.device
                ) # (batch_size, n_classes, patch_size, patch_size)

                loss = self.criterion(logits_batch, target_batch)
                losses.append(loss.item())

                for logits_patch in logits_batch:
                    logits_patches.append(logits_patch)

        val_loss = sum(losses) / len(losses)

        logits_patches = logits_patches[:init_n_patches]
        logits = combine_from_patches(
            logits_patches,
            patch_size=PATCH_SIZE,
            offset=OFFSET,
            overlap=PATCH_OVERLAP,
            src_shape=inputs.shape[-2:],
            device=self.device
        ) # (n_classes, H, W) in GPU memory
        target = target.to(device=self.device)
    
        """ Prepare for metrics calculation """
        if OFFSET > 0:
            logits_cropped = logits[..., OFFSET: -OFFSET, OFFSET: -OFFSET]
            target_cropped = target[..., OFFSET: -OFFSET, OFFSET: -OFFSET]

        """ Metrics for output activation """
        activated = F.softmax(input=logits_cropped, dim=1) # (n_classes, H, W)
        iou_activated_per_class = iou_per_class(target_cropped, activated)

        """ Metrics for prediction """
        prediction_mask = torch.argmax(activated, dim=0) # (H, W)
        prediction = one_hot(prediction_mask, n_classes=activated.shape[0])
        iou_pred_per_class = iou_per_class(target_cropped, prediction)

        eval_res = EvaluationResult(
            iou_activated_per_class=iou_activated_per_class,
            iou_pred_per_class=iou_pred_per_class,
            accuracy = accuracy(target_cropped, prediction)
        )
        return val_loss, eval_res


    def _epoch_end(
        self, epoch: int, train_losses: list[float], val_losses: list[float]
    ) -> dict[str, float]:
        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        """ Print training/validation losses """
        print(f"[Epoch: {epoch + 1}] Training loss: {avg_train_loss:.3f} Validation loss: {avg_val_loss:.3f}")

        """ Reduce lr on plateua """
        self.scheduler.step(avg_val_loss)

        """ Save best model """
        if avg_val_loss < self.best_avg_val_loss:
            self.best_avg_val_loss = avg_val_loss
            self.best_model = copy.deepcopy(self.model)

        return {
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "lr": self.scheduler.get_last_lr()
        }


    def fit(self):
        self.model.to(self.device)
        epoch_outputs = []
        for epoch in range(self.max_epochs):
            train_dataloader, val_dataloader = self._split_dataset()

            self.model.train()
            train_losses = []
            bar = tqdm(train_dataloader, postfix={"train_loss": 0.0})
            for inputs, target in bar:
                train_losses.append(self._train_image(inputs, target))
                bar.set_postfix(ordered_dict={"train_loss": train_losses[-1]})

            self.model.eval()
            val_losses = []
            bar = tqdm(val_dataloader, postfix={"val_loss": 0.0})
            for inputs, target in bar:
                val_loss, _ = self._validate_image(inputs, target)
                val_losses.append(val_loss)
                bar.set_postfix(ordered_dict={"val_loss": val_loss})

            self._epoch_end(epoch, train_losses, val_losses)

        # save_training_outputs(epoch_outputs, self.exp_path)
        torch.save(self.best_model.state_dict(), Path.cwd() / self.exp_path / "best_model.pth")


    def test(self, description):
        if description == "test":
            self.model.load_state_dict(torch.load(self.exp_path / "best_model.pth"))

        self.model.to(self.device)
        test_dataloader = DataLoader(self._test_dataset, batch_size=1, num_workers=4)
        
        self.model.eval()
        eval_results= []
        bar = tqdm(test_dataloader)
        for inputs, target in bar:
            _, eval_res = self._validate_image(inputs, target)
            eval_results.append(eval_res)
            write_metrics(self.log_detailed, eval_results[-1], description=f"{description}, image {len(eval_results)}")
            
        total_eval_res = evaluate_dataset(eval_results)
        write_metrics(self.log, total_eval_res, description=f"{description}, total")
