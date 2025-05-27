"""
This file contains all dataset classes for the EMBED dataset.

IMPORTANT: Pre-requisites.
1. Download the dataset, unzip and update the corresponding
path in the default_paths.py file
2. Run data_handling/preprocess_and_splits_creation/1-generate-splits/mammo_df_creation.ipynb
to save our train/val/split files.
"""

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
from skimage import io
from torch.utils.data import Dataset

from data_handling.base import BaseDataModuleClass
from data_handling.caching import SharedCache
from torchvision.transforms import Resize

from default_paths import ROOT, EMBED_ROOT

domain_maps = {
    "HOLOGIC, Inc.": 0,
    "GE MEDICAL SYSTEMS": 1,
    "FUJIFILM Corporation": 2,
    "GE HEALTHCARE": 3,
    "Lorad, A Hologic Company": 4,
}

tissue_maps = {"A": 0, "B": 1, "C": 2, "D": 3}
modelname_map = {
    "Selenia Dimensions": 0,
    "Senographe Essential VERSION ADS_53.40": 5,
    "Senographe Essential VERSION ADS_54.10": 5,
    "Senograph 2000D ADS_17.4.5": 2,
    "Senograph 2000D ADS_17.5": 2,
    "Lorad Selenia": 3,
    "Clearview CSm": 4,
    "Senographe Pristina": 1,
}


def preprocess_breast(image_path, target_size):
    """
    Loads the image performs basic background removal around the breast.
    Works for text but not for objects in contact with the breast (as it keeps the
    largest non background connected component.)
    """
    image = cv2.imread(str(image_path))

    if image is None:
        # sometimes bug in reading images with cv2
        from skimage.util import img_as_ubyte

        image = io.imread(image_path)
        gray = img_as_ubyte(image.astype(np.uint16))
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.threshold(gray, 5, 255, cv2.THRESH_BINARY)[1]

    # Connected components with stats.
    nb_components, output, stats, _ = cv2.connectedComponentsWithStats(
        thresh, connectivity=4
    )

    # Find the largest non background component.
    # Note: range() starts from 1 since 0 is the background label.
    max_label, _ = max(
        [(i, stats[i, cv2.CC_STAT_AREA]) for i in range(1, nb_components)],
        key=lambda x: x[1],
    )
    mask = output == max_label
    img = torch.tensor((gray * mask) / 255.0).unsqueeze(0).float()
    img = Resize(target_size, antialias=True)(img)
    return img


class EmbedDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        transform: torch.nn.Module,
        target_size=[224, 192],
        cache: bool = True,
    ) -> None:
        self.imgs_paths = df.image_path.values
        self.shortpaths = df.shortimgpath.values
        self.labels = df.tissueden.values
        self.num_classes = len(np.unique(self.labels))

        self.transform = transform
        self.target_size = target_size
        self.views = df.ViewLabel.values
        self.scanner = df.SimpleModelLabel.values
        self.cview = df.FinalImageType.apply(lambda x: 0 if x == "2D" else 1).values
        self.age = df.age_at_study.values
        self.densities = df.tissueden.values

        data_dims = [1, self.target_size[0], self.target_size[1]]
        if cache:
            self.cache = SharedCache(
                size_limit_gib=96,
                dataset_len=self.labels.shape[0],
                data_dims=data_dims,
                dtype=torch.float32,
            )
        else:
            self.cache = None

    def __getitem__(self, index) -> Any:
        if self.cache is not None:
            # retrieve data from cache if it's there
            img = self.cache.get_slot(index)
            # x will be None if the cache slot was empty or OOB
            if img is None:
                img = preprocess_breast(self.imgs_paths[index], self.target_size)
                self.cache.set_slot(index, img, allow_overwrite=True)  # try to cache x
        else:
            img = preprocess_breast(self.imgs_paths[index], self.target_size)
        sample = {}
        age = self.age[index]
        sample["cview"] = self.cview[index]
        sample["shortpath"] = str(self.shortpaths[index])
        sample["real_age"] = age
        sample["view"] = self.views[index]
        sample["density"] = torch.nn.functional.one_hot(
            torch.tensor(self.densities[index]).long(), num_classes=4
        ).detach()
        sample["y"] = self.labels[index]
        sample["scanner_int"] = self.scanner[index]
        sample["scanner"] = torch.nn.functional.one_hot(
            torch.tensor(self.scanner[index]).long(), num_classes=6
        ).detach()

        img = self.transform(img).float()

        sample["x"] = img

        return sample

    def __len__(self):
        return self.labels.shape[0]


class EmbedDataModule(BaseDataModuleClass):
    @property
    def dataset_name(self) -> str:
        return "EMBED"

    def create_datasets(self) -> None:
        train_dataset = pd.read_csv(ROOT / "experiments" / "train_embed.csv")
        val_dataset = pd.read_csv(ROOT / "experiments" / "val_embed.csv")
        test_dataset = pd.read_csv(ROOT / "experiments" / "test_embed.csv")
        self.target_size = self.config.data.augmentations.resize

        self.dataset_train = EmbedDataset(
            df=train_dataset,
            transform=self.train_tsfm,
            target_size=self.target_size,
            cache=self.config.data.cache,
        )

        self.dataset_val = EmbedDataset(
            df=val_dataset,
            transform=self.val_tsfm,
            target_size=self.target_size,
            cache=self.config.data.cache,
        )

        self.dataset_test = EmbedDataset(
            df=test_dataset,
            transform=self.val_tsfm,
            target_size=self.target_size,
            cache=True,
        )

    @property
    def num_classes(self) -> int:
        return 4
