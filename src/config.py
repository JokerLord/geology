import albumentations as A
from albumentations.pytorch import ToTensorV2

from pathlib import Path


# LUMENSTONE_PATH = Path.home() / "dev_data" / "LumenStone" / "S1_v1" # on local machine
LUMENSTONE_PATH = Path.cwd().parent / "LumenStone" / "S1_v1" # on sinope server

CLASS_NAMES = [
    "BG",
    "Ccp",
    "Gl",
    "Mag",
    "Brt",
    "Po",
    "Py/Mrc",
    "Pn",
    "Sph",
    "Apy",
    "Hem",
    "Tnt/Ttr",
    "Kvl",
]

MISSED_CLASS_CODES = [3, 5, 7, 9, 10, 12]

TRAIN_TRANSFORM = A.Compose([ToTensorV2(transpose_mask=True)])

VAL_TRANSFORM = A.Compose([ToTensorV2(transpose_mask=True)])

N_FILTERS = 16

PATCH_SIZE = 384
PATCH_OVERLAP = 0.5
OFFSET = 8

BATCH_SIZE = 16
SPLIT_RATIO = 0.85

LR = 1e-4
