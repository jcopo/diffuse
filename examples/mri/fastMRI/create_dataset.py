from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import numpy as np
import glob
import os
import h5py
import sigpy as sp


def numpy_collate(batch):
    return np.asarray(data.default_collate(batch))


class FastMRIDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        min_slice: int = 13,
        max_slice: int = 29,
        image_size: int = 320,
    ):
        self.data_path = data_path
        self.min_slice = min_slice
        self.max_slice = max_slice
        self.image_size = image_size
        self.file_list = glob.glob(os.path.join(data_path, "*.h5"))
        self.num_slices = []
        for file in self.file_list:
            with h5py.File(file, "r") as f:
                self.num_slices.append(
                    f["kspace"][self.min_slice : self.max_slice].shape[0]
                )
        self.slice_mapper = np.cumsum(self.num_slices) - 1

    def __len__(self):
        return sum(self.num_slices)

    def __getitem__(self, idx):
        file_idx = np.searchsorted(self.slice_mapper, idx)
        slice_idx = idx - self.slice_mapper[file_idx] + self.num_slices[file_idx] - 1
        with h5py.File(self.file_list[file_idx], "r") as f:
            gt_ksp = f["kspace"][self.min_slice : self.max_slice][slice_idx]
        gt_ksp = sp.resize(gt_ksp, (gt_ksp.shape[0], self.image_size))
        gt_ksp = sp.ifft(gt_ksp, axes=(-2,))
        gt_ksp = sp.resize(gt_ksp, (self.image_size, gt_ksp.shape[1]))
        gt_ksp = sp.fft(gt_ksp, axes=(-2,))
        gt_xsp = np.fft.fftshift(
            np.fft.ifft2(
                np.fft.ifftshift(gt_ksp, axes=[-2, -1]), norm="ortho", axes=[-2, -1]
            ),
            axes=[-2, -1],
        )
        gt_xsp_scale_factor = np.percentile(np.abs(gt_xsp), 99)
        gt_xsp /= gt_xsp_scale_factor
        gt_xsp = np.stack([np.real(gt_xsp), np.imag(gt_xsp)], axis=-1)
        return gt_xsp


def get_dataloader(cfg, train: bool = True):
    dataset = FastMRIDataset(
        cfg["path_dataset"]
    )
    return DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
        collate_fn=numpy_collate,
    )
