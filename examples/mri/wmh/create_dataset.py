import os


import numpy as np

import pandas as pd
import h5py
import glob
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

import torchio as tio
import sigpy as sp


def numpy_collate(batch):
    return np.asarray(data.default_collate(batch))


def create_h5(cfg):
    path_dataset = cfg["path_dataset"]
    save_path_h5 = cfg["save_path_h5"]
    csv = pd.read_csv(os.path.join(path_dataset, "data.csv"), sep=";")
    for _, sub in csv.iterrows():
        path_img = os.path.join(
            path_dataset, sub.Path, str(sub.ID), f"pre/mni_FLAIR.nii.gz"
        )
        path_mask = os.path.join(path_dataset, sub.Path, str(sub.ID), "mni_wmh.nii.gz")

        subject_dict = {
            "vol": tio.ScalarImage(path_img),
            "mask": tio.LabelMap(path_mask),
            "ID": sub.ID,
        }

        subject = tio.Subject(subject_dict)

        with h5py.File(os.path.join(save_path_h5, f"{sub.ID}.h5"), "w") as f:
            f.create_dataset("volume", data=subject.vol.numpy().squeeze(0))
            f.create_dataset("mask", data=subject.mask.numpy().squeeze(0))


class WMHDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        min_slice: int = 26,
        max_slice: int = 72,
    ):
        self.data_path = data_path
        self.min_slice = min_slice
        self.max_slice = max_slice
        self.file_list = glob.glob(os.path.join(data_path, "*.h5"))
        self.num_slices = []
        self.cached_data = {}
        for file in self.file_list:
            with h5py.File(file, "r") as f:
                self.cached_data[file] = {
                    "volume": f["volume"][..., self.min_slice : self.max_slice],
                    "mask": f["mask"][..., self.min_slice : self.max_slice],
                }

        self.num_slices = [v["volume"].shape[-1] for v in self.cached_data.values()]
        self.slice_mapper = np.cumsum(self.num_slices) - 1

    def __len__(self):
        return sum(self.num_slices)

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.slice_mapper, idx)
        slice_idx = idx - self.slice_mapper[file_idx] + self.num_slices[file_idx] - 1

        data = self.cached_data[self.file_list[file_idx]]
        vol = data["volume"][..., slice_idx]
        mask = data["mask"][..., slice_idx]

        vol = sp.resize(vol, (112, 112))
        mask = sp.resize(mask, (112, 112))
        vol_ksp = np.fft.fft2(vol, norm="ortho", axes=[-2, -1])
        vol_xsp = np.fft.ifft2(vol_ksp, norm="ortho", axes=[-2, -1])
        vol_xsp_scale_factor = np.percentile(np.abs(vol_xsp), 99)
        vol_xsp /= vol_xsp_scale_factor
        vol_xsp = np.stack([np.real(vol_xsp), np.imag(vol_xsp), mask], axis=-1)
        return vol_xsp


def get_dataloader(cfg, train: bool = True):
    folder = "train_data" if train else "val_data"
    path_dataset = os.path.join(cfg["path_dataset"], folder)
    dataset = WMHDataset(
        path_dataset,
    )
    return DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=train,
        num_workers=cfg["training"]["num_workers"],
        collate_fn=numpy_collate,
        drop_last=True,
    )