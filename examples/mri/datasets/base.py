from torch.utils import data
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Tuple, Any, Union
import numpy as np
import glob
import os


def numpy_collate(batch: List[Any]) -> np.ndarray:
    return np.asarray(data.default_collate(batch))


class BaseMRIDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[callable] = None) -> None:
        self.data_path = data_path
        self.transform = transform
        self.file_list: List[str] = glob.glob(os.path.join(data_path, "*.h5"))
        self.num_slices: List[int] = []
        self.cached_data: Dict[str, Any] = {}

        self._cache_data()

    def _cache_data(self) -> None:
        """Override this method to define how data should be cached"""
        raise NotImplementedError

    def __len__(self) -> int:
        return sum(self.num_slices)

    def __getitem__(self, idx: int) -> Any:
        """Override this method to define how data should be accessed"""
        raise NotImplementedError


def get_base_dataloader(
    dataset: Dataset, cfg: Dict[str, Any], train: bool = True, latent: bool = False
) -> Union[Tuple[DataLoader, DataLoader], DataLoader]:
    """Common dataloader creation logic"""
    if not latent:
        config_key = "score"
    else:
        config_key = "latent"

    train_ratio = cfg["training"][config_key].get("train_ratio", 0.8)
    batch_size = cfg["training"][config_key].get("batch_size", 32)
    num_workers = cfg["training"][config_key].get("num_workers", 0)

    if train:
        total_size = len(dataset)
        train_size = int(total_size * train_ratio)
        val_size = total_size - train_size

        train_dataset, val_dataset = data.random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            drop_last=True,
        )

        return train_loader, val_loader

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=numpy_collate,
        drop_last=True,
    )
