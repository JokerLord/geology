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

CLASS_COLORS = [
    '#000000',
    '#ff0000',
    '#cbff00',
    '#00ff66',
    '#0065ff',
    '#cc00ff',
    '#ff4c4c',
    '#dbff4c',
    '#4cff93',
    '#4c93ff',
    '#db4cff',
    '#ff9999',
    '#eaff99',
]

MISSED_CLASS_CODES = [3, 5, 7, 9, 10, 12]

present_class_codes = [code for code in range(len(CLASS_NAMES)) if code not in MISSED_CLASS_CODES]
codes2squeezed_codes = {code: i for i, code in enumerate(present_class_codes)}
squeezed_codes2labels = {i: CLASS_NAMES[code] for i, code in enumerate(present_class_codes)}
labels2colors = {class_name: CLASS_COLORS[i] for i, class_name in enumerate(CLASS_NAMES)}

TRAIN_TRANSFORM = A.Compose([ToTensorV2(transpose_mask=True)])

VAL_TRANSFORM = A.Compose([ToTensorV2(transpose_mask=True)])

N_FILTERS = 16

PATCH_SIZE = 384
PATCH_OVERLAP = 0.5
OFFSET = 8

BATCH_SIZE = 16
SPLIT_RATIO = 0.8

LR = 1e-4
