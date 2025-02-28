from .base import BaseMRIDataset
import h5py
import numpy as np
import sigpy as sp


# TODO: Add latent space return option


class WMHDataset(BaseMRIDataset):
    def _cache_data(self) -> None:
        for file in self.file_list:
            with h5py.File(file, "r") as f:
                self.cached_data[file] = {
                    "volume": f["volume"][..., self.min_slice : self.max_slice],
                    "mask": f["mask"][..., self.min_slice : self.max_slice],
                }
        self.num_slices = [v["volume"].shape[-1] for v in self.cached_data.values()]
        self.slice_mapper = np.cumsum(self.num_slices) - 1

    def __getitem__(self, idx: int) -> np.ndarray:
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
