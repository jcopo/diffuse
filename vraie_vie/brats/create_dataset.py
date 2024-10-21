import numpy as np
from torch.utils import data

import torchio as tio

import os


def numpy_collate(batch):
    collated_batch = data.default_collate(batch)
    return {key: np.asarray(value) for key, value in collated_batch.items()}


def get_transform():
    transform = tio.Compose(
        [
            tio.CropOrPad((160, 240, 155)),
            tio.RandomFlip(axes=1),
            tio.RandomElasticDeformation(),
            tio.RandomGamma(),
            tio.RescaleIntensity((0, 1), percentiles=(1, 99)),
        ]
    )
    return transform


def create_dataset(cfg):
    path_dataset = cfg["path_dataset"]
    path_subjects = [e for e in os.listdir(path_dataset) if "BraTS-GLI" in e]

    subjects = []
    for id_subject in path_subjects:
        path_img = os.path.join(path_dataset, id_subject, f"{id_subject}-t1n.nii.gz")

        subject_dict = {
            "vol": tio.ScalarImage(path_img),
            "ID": id_subject,
            "path": path_img,
        }

        subject = tio.Subject(subject_dict)
        subjects.append(subject)

    return tio.SubjectsDataset(subjects, transform=get_transform())


def get_train_dataloader(cfg):
    dataset = create_dataset(cfg)
    sampler = tio.UniformSampler((160, 240, 1))
    queue = tio.Queue(
        dataset, max_length=100, samples_per_volume=5, sampler=sampler, num_workers=0
    )
    patches_loader = tio.SubjectsLoader(
        queue, batch_size=cfg["batch_size"], num_workers=0, collate_fn=numpy_collate
    )
    return patches_loader
