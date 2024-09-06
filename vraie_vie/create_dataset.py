import os

from jax.tree_util import tree_map

import numpy as np

import pandas as pd

import torch
import torch.nn.functional as F
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

import torchio as tio


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


def get_transform(cfg):
    transform = tio.Compose(
        [
            tio.RescaleIntensity(
                (0, 1), percentiles=(cfg.get("perc_low", 1), cfg.get("perc_high", 99))
            ),
        ]
    )
    return transform


def load_nifti(cfg, type="Training"):
    path_dataset = cfg["path_dataset"]

    csv = pd.read_csv(os.path.join(path_dataset, "data.csv"), sep=";")

    subjects = []
    for _, sub in csv.iterrows():
        if sub.Type == type:
            path_img = os.path.join(
                path_dataset, sub.Path, str(sub.ID), f"pre/mni_{cfg['modality']}.nii.gz"
            )
            path_mask = os.path.join(
                path_dataset, sub.Path, str(sub.ID), f"mni_wmh.nii.gz"
            )

            subject_dict = {
                "vol": tio.ScalarImage(path_img),
                "mask": tio.LabelMap(path_mask) if type == "Training" else None,
                "center": sub.Center,
                "ID": sub.ID,
                "path": sub.Path,
            }

            subject = tio.Subject(subject_dict)
            subjects.append(subject)

    return tio.SubjectsDataset(subjects, transform=get_transform(cfg))


class vol2slice(Dataset):
    def __init__(self, cfg, ds, type):
        self.ds = ds
        self.cfg = cfg
        self.type = type

    def __len__(self):
        return len(self.ds) * self.cfg["slice_size_template"]

    def __getitem__(self, idx):
        idx_subject = idx // self.cfg["slice_size_template"]
        idx_slice = idx % self.cfg["slice_size_template"] + self.cfg["begin_slice"]

        subject = self.ds.__getitem__(idx_subject)

        subject["vol"][tio.DATA] = subject["vol"][tio.DATA][0, ..., idx_slice]

        if self.type == "Training":
            subject["mask"][tio.DATA] = subject["mask"][tio.DATA][0, ..., idx_slice]

            data_masked = torch.concatenate(
                [
                    subject["vol"][tio.DATA][..., None],
                    subject["mask"][tio.DATA][..., None].type(torch.float32),
                ],
                dim=-1,
            )

            padding = (0, 0, 0, 3, 0, 1)
            padded_tensor = F.pad(data_masked, padding, "constant", 0)

            return padded_tensor

        return subject["vol"][tio.DATA]


def create_dataset(cfg, type="Training"):
    ds = load_nifti(cfg, type)
    ds = vol2slice(cfg, ds, type)
    return ds


class WMH:
    def __init__(self, cfg):
        self.cfg = cfg

    def setup(self):
        self.train_dataset = create_dataset(self.cfg, type="Training")
        self.test_dataset = create_dataset(self.cfg, type="Test")

    def get_train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
            drop_last=True,
            collate_fn=numpy_collate,
        )

    def get_test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.cfg["batch_size"],
            shuffle=False,
            num_workers=self.cfg["num_workers"],
            pin_memory=True,
            collate_fn=numpy_collate,
        )


if __name__ == "__main__":
    config = {
        "modality": "FLAIR",
        "slice_size_template": 91,
        "path_dataset": "/Users/geoffroyoudoumanessah/Documents/these/projects/datasets/WMH",
        "batch_size": 32,
        "num_workers": 0,
    }

    wmh = WMH(config)
    wmh.setup()

    train_loader = wmh.get_train_dataloader()
