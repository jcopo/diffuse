from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import h5py
import sigpy as sp

def numpy_collate(batch):
    return np.asarray(data.default_collate(batch))

class BrainfastMRIDataset(Dataset):
    def __init__(self, data_path: str, image_size: int = 384):
        self.data_path = data_path
        self.image_size = image_size
        self.file_list = glob.glob(os.path.join(data_path, "*.h5"))
        self.num_slices = []

        self.cached_data = {}
        for file in self.file_list:
            with h5py.File(file, "r") as f:
                self.cached_data[file] = {
                    "kspace": np.array(f["kspace"])
                }

        self.num_slices = [v["kspace"].shape[0] for v in self.cached_data.values()]
        self.slice_mapper = np.cumsum(self.num_slices) - 1

    def __len__(self):
        return sum(self.num_slices)

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.slice_mapper, idx)
        slice_idx = idx - self.slice_mapper[file_idx] + self.num_slices[file_idx] - 1
        
        data = self.cached_data[self.file_list[file_idx]]
        return data["kspace"][slice_idx]


def get_dataloader(cfg, train: bool = True):
    folder = "train_data" if train else "val_data"
    path_dataset = os.path.join(cfg["path_dataset"], folder)
    dataset = BrainfastMRIDataset(
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