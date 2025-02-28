from .base import BaseMRIDataset
import h5py
import numpy as np
from diffuse.neural_network.autoencoderKL import AutoencoderKL
from ..training.utils import load_best_model_checkpoint
import jax
from tqdm import tqdm


class KneeFastMRIDataset(BaseMRIDataset):
    def _cache_data(self) -> None:
        if self.return_latent:
            _, params, _ = load_best_model_checkpoint(self.cfg, self.return_latent)
            model = AutoencoderKL(**self.cfg["neural_network"]["autoencoder"])
            for file in tqdm(self.file_list, desc="Caching data and encoding"):
                with h5py.File(file, "r") as f:
                    data = np.array(f["xspace"])
                    self.key, subkey = jax.random.split(self.key)
                    posterior, _ = model.apply(
                        params, data, deterministic=True, rngs=subkey
                    )
                    latent = posterior.mode()
                    self.cached_data[file] = {"xspace": np.array(latent)}
        else:
            for file in tqdm(self.file_list, desc="Caching data"):
                with h5py.File(file, "r") as f:
                    self.cached_data[file] = {"xspace": np.array(f["xspace"])}
        self.num_slices = [v["xspace"].shape[0] for v in self.cached_data.values()]
        self.slice_mapper = np.cumsum(self.num_slices) - 1

    def __getitem__(self, idx: int) -> np.ndarray:
        file_idx = np.searchsorted(self.slice_mapper, idx)
        slice_idx = idx - self.slice_mapper[file_idx] + self.num_slices[file_idx] - 1

        data = self.cached_data[self.file_list[file_idx]]
        return data["xspace"][slice_idx]
