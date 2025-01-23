import numpy as np
from torch.utils import data
from torch.utils.data import Dataset, DataLoader
import glob
import h5py
import torchio as tio
from tqdm import tqdm

import os


def numpy_collate(batch):
    return np.asarray(data.default_collate(batch))


def create_h5(cfg):
    path_dataset = cfg["path_dataset"]
    path_subjects = [e for e in os.listdir(path_dataset) if "BraTS-GLI" in e]
    save_path_h5 = cfg["save_path_h5"]
    transform = tio.CropOrPad((160, 240, 155))
    for id_subject in tqdm(path_subjects):
        path_img = os.path.join(path_dataset, id_subject, f"{id_subject}-t1n.nii.gz")
        path_mask = os.path.join(
            path_dataset, id_subject, f"{id_subject}-mask-unhealthy.nii.gz"
        )

        subject_dict = {
            "vol": tio.ScalarImage(path_img),
            "mask": tio.LabelMap(path_mask),
            "ID": id_subject,
            "path": path_img,
        }

        subject = tio.Subject(subject_dict)
        subject = transform(subject)

        with h5py.File(os.path.join(save_path_h5, f"{id_subject}.h5"), "w") as f:
            f.create_dataset("volume", data=subject.vol.numpy().squeeze(0))
            f.create_dataset("mask", data=subject.mask.numpy().squeeze(0))


class BratsDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        min_slice: int = 50,  # 46,
        max_slice: int = 120,  # 130,
    ):
        self.data_path = data_path
        self.min_slice = min_slice
        self.max_slice = max_slice
        self.file_list = glob.glob(os.path.join(data_path, "*.h5"))
        self.num_slices = []

        self.cached_data = {}
        for file in tqdm(self.file_list):
            with h5py.File(file, "r") as f:
                self.cached_data[file] = {
                    "volume": f["volume"][..., min_slice:max_slice],
                    "mask": f["mask"][..., min_slice:max_slice],
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

        vol_ksp = np.fft.fft2(vol, norm="ortho", axes=[-2, -1])
        vol_xsp = np.fft.ifft2(vol_ksp, norm="ortho", axes=[-2, -1])
        vol_xsp_scale_factor = np.percentile(np.abs(vol_xsp), 99)
        vol_xsp /= vol_xsp_scale_factor
        vol_xsp = np.stack([np.real(vol_xsp), np.imag(vol_xsp), mask], axis=-1)
        return vol_xsp


def get_dataloader(cfg, train: bool = True):
    folder = "train_data" if train else "val_data"
    path_dataset = os.path.join(cfg["path_dataset"], folder)
    dataset = BratsDataset(
        path_dataset,
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
