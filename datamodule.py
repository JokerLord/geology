import pytorch_lightning as pl

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import Subset, DataLoader

from sklearn.model_selection import train_test_split

from config import BATCH_SIZE, TRAIN_TRANSFORM, VAL_TRANSFORM, SPLIT_RATIO
from dataset import LumenStoneDataset


class LumenStoneDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root_dir,
        batch_size=BATCH_SIZE,
        split_ratio=SPLIT_RATIO,
        train_transform=TRAIN_TRANSFORM,
        val_transform=VAL_TRANSFORM,
    ):
        """
        Arguments:
            root_dir (string): Directory with all the images.
            batch_size (int, optional): Batch size. Default: 16.
            split_ration (float, optional): Train/validation split ration. Default: 0.85.
            train_transform (callable, optional): Optional transform to be applied
                on train samples.
            val_transform (callable, optional): Optinal transofrm
        """

        self.root_dir = root_dir
        self.batch_size = batch_size
        self.split_ratio = split_ratio
        self.train_transform = train_transform
        self.val_transform = val_transform

    def setup(self, stage):

        print("!!!!!!SETUP!!!!!!")

        """ Assign train/val datasets for use in dataloaders """
        if stage == "fit":
            lumenstone_train = LumenStoneDataset(
                root_dir=self.root_dir, train=True, transform=TRAIN_TRANSFORM
            )
            lumenstone_val = LumenStoneDataset(
                root_dir=self.root_dir, train=True, transform=VAL_TRANSFORM
            )

            indices = list(range(len(lumenstone_train))) 
            train_indices, val_indices = train_test_split(indices, train_size=self.split_ratio)
            self.lumenstone_train = Subset(lumenstone_train, train_indices)
            self.lumenstone_val = Subset(lumenstone_val, val_indices)

        """ Assign test dataset for use in dataloader(s) """
        if stage == "test":
            self.lumenstone_test = LumenStoneDataset(
                self.root_dir, train=False, transform=VAL_TRANSFORM
            )

        if stage == "predict":
            self.lumenstone_predict = LumenStoneDataset(
                self.root_dir, train=False, transform=VAL_TRANSFORM
            )

    def train_dataloader(self):
        return DataLoader(self.lumenstone_train, batch_size=self.batch_size)
    
    def val_dataloader(self):
        return DataLoader(self.lumenstone_val, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.lumenstone_test, batch_size=self.batch_size)
    
    def predict_dataloader(self):
        return DataLoader(self.lumenstone_predict, batch_size=self.batch_size)
