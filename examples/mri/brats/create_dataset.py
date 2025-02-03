import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import glob
import h5py

import os


def numpy_collate(batch):
    return np.asarray(data.default_collate(batch))


class BratsDataset(Dataset):
    def __init__(
        self,
        data_path: str,
    ):
        self.data_path = data_path
        self.file_list = glob.glob(os.path.join(data_path, "*.h5"))
        self.num_slices = []

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        data = h5py.File(self.file_list[idx], "r")
        vol = data["image"][..., -1]
        mask = data["mask"].sum(axis=-1)

        vol_scale_factor = np.percentile(vol, 99)
        vol /= vol_scale_factor
        vol = np.stack([vol, mask], axis=-1)
        return vol


def get_dataloader(cfg, train: bool = True):
    folder = "train_data" if train else "val_data"
    path_dataset = os.path.join(cfg["path_dataset"], folder)
    dataset = BratsDataset(
        path_dataset,
    )

    if train:
        # Get train/val split ratio from config, default to 0.8
        train_ratio = cfg["training"].get("train_ratio", 0.8)

        # Calculate split sizes
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size

        # Split dataset
        train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

        # Create and return both dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=cfg["training"]["num_workers"],
            collate_fn=numpy_collate,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            num_workers=cfg["training"]["num_workers"],
            collate_fn=numpy_collate,
            drop_last=True,
        )

        return train_loader, val_loader

    # If train=False, return single test dataloader as before
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=numpy_collate,
        drop_last=True,
    )