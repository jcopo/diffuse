import os

from dipy.align.imaffine import MutualInformationMetric, AffineRegistration

from dipy.align.transforms import (
    TranslationTransform3D,
    RigidTransform3D,
    AffineTransform3D,
)

from dipy.io.image import load_nifti, save_nifti

import numpy as np

import pandas as pd

from tqdm import tqdm


def check_and_correct_affine(affine):
    rotation = affine[:3, :3]
    if not np.allclose(np.dot(rotation, rotation.T), np.eye(3), atol=1e-3):
        u, _, vh = np.linalg.svd(rotation, full_matrices=False)
        corrected_rotation = np.dot(u, vh)
        affine[:3, :3] = corrected_rotation
    return affine


def coregister(cfg):
    path_dataset = cfg["path_dataset"]
    csv = pd.read_csv(os.path.join(path_dataset, "data.csv"), sep=";")

    path_template = cfg["flair_template_path"]
    template_data, template_affine = load_nifti(path_template)

    for _, sub in tqdm(csv.iterrows()):
        path_img = os.path.join(
            path_dataset, sub.Path, str(sub.ID), f"pre/{cfg['modality']}.nii.gz"
        )
        moving_data, moving_affine = load_nifti(path_img)

        path_mask = os.path.join(path_dataset, sub.Path, str(sub.ID), f"wmh.nii.gz")
        mask_data, _ = load_nifti(path_mask)

        nbins = 32
        sampling_prop = None
        metric = MutualInformationMetric(nbins, sampling_prop)

        level_iters = [10, 10, 5]
        sigmas = [3.0, 1.0, 0.0]
        factors = [4, 2, 1]

        affreg = AffineRegistration(
            metric=metric, level_iters=level_iters, sigmas=sigmas, factors=factors
        )

        transform = TranslationTransform3D()
        params0 = None
        translation = affreg.optimize(
            template_data,
            moving_data,
            transform,
            params0,
            template_affine,
            moving_affine,
        )

        transform = RigidTransform3D()
        rigid = affreg.optimize(
            template_data,
            moving_data,
            transform,
            params0,
            template_affine,
            moving_affine,
            starting_affine=translation.affine,
        )

        transform = AffineTransform3D()

        affreg.level_iters = [1000, 1000, 100]
        affine = affreg.optimize(
            template_data,
            moving_data,
            transform,
            params0,
            template_affine,
            moving_affine,
            starting_affine=rigid.affine,
        )

        transformed = affine.transform(moving_data)
        transformed_mask = affine.transform(mask_data)

        resulting_affine = check_and_correct_affine(affine.affine)

        save_path_data = os.path.join(
            path_dataset, sub.Path, str(sub.ID), f"pre/mni_{cfg['modality']}.nii.gz"
        )
        save_path_mask = os.path.join(
            path_dataset, sub.Path, str(sub.ID), f"mni_wmh.nii.gz"
        )

        save_nifti(save_path_data, transformed.astype(np.float32), resulting_affine)
        save_nifti(
            save_path_mask, (transformed_mask > 0).astype(np.uint8), resulting_affine
        )


if __name__ == "__main__":
    cfg = {
        "modality": "FLAIR",
        "flair_template_path": "/Users/geoffroyoudoumanessah/Documents/these/projects/datasets/WMH/MNI-FLAIR-2.0mm.nii.gz",
        "path_dataset": "/Users/geoffroyoudoumanessah/Documents/these/projects/datasets/WMH",
    }

    coregister(cfg)
