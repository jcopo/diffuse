# Standard library imports
from typing import Tuple, Union
import os

# Third-party imports
from torch.utils.data import DataLoader

# Local imports
from .base import get_base_dataloader
from .knee_fastmri import KneeFastMRIDataset
from .brain_fastmri import BrainFastMRIDataset
from .wmh import WMHDataset

# Dataset registry
datasets_zoo = {
    "kneeFastMRI": KneeFastMRIDataset,
    "brainFastMRI": BrainFastMRIDataset,
    "WMH": WMHDataset,
}


def get_dataloader(
    cfg, train: bool = True, model: str = "score"
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    return_latent = model == "score"
    dataset = datasets_zoo[cfg["dataset"]](cfg, train, return_latent)
    return get_base_dataloader(dataset, cfg, train, model)
