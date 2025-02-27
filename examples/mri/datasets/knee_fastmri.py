from .base import BaseMRIDataset
import h5py
import numpy as np


class KneeFastMRIDataset(BaseMRIDataset):
    def _cache_data(self) -> None:
        for file in self.file_list:
            with h5py.File(file, "r") as f:
                self.cached_data[file] = {"xspace": np.array(f["xspace"])}
        self.num_slices = [v["xspace"].shape[0] for v in self.cached_data.values()]
        self.slice_mapper = np.cumsum(self.num_slices) - 1

    def __getitem__(self, idx: int) -> np.ndarray:
        file_idx = np.searchsorted(self.slice_mapper, idx)
        slice_idx = idx - self.slice_mapper[file_idx] + self.num_slices[file_idx] - 1

        data = self.cached_data[self.file_list[file_idx]]
        return data["xspace"][slice_idx]
