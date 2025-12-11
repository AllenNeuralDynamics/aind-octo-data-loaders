"""
Masked Zarr DataLoader for volumetric data with cloud coverage filtering.
Reads zarr v2 and v3 formats.
"""

import dask.array as da
import numpy as np
import pandas as pd
import random
import torch.distributed as dist
import xarray as xr
import xbatcher

from torch.utils.data import Dataset, IterableDataset
from tqdm import tqdm
from aind_large_scale_prediction.io import OMEZarrReader


class ZarrDataset(Dataset):
    def __init__(
        self,
        sample,
        zarr_file,
        scale,
        volume_size=128,
        do_pad_array=True,
        platform=None,
        overlap_pct=0.0,
        zarr_version="2.0",
        transform=None
    ):
        self.sample = sample
        self.zarr_file = zarr_file
        self.scale = scale
        self.volume_size = volume_size
        self.do_pad_array = do_pad_array
        self.platform = platform
        self.overlap_pct = overlap_pct
        self.zarr_version = zarr_version
        self.transform = transform

        self.datasets = dict()

        # Loading zarr v2 or v3
        self.load_and_pad_zarr(
            zarr_file=self.zarr_file,
            scale=self.scale,
            dataset_name="volume",
            zarr_version=self.zarr_version
        )
        self._is_initialized = False

        self._lazy_init_generator()

    def pad_dask_array(self, dask_array):
        if len(dask_array.shape) == 5:
            _, _, z, y, x = dask_array.shape
        elif len(dask_array.shape) == 3:
            z, y, x = dask_array.shape
        else:
            raise ValueError(
                f"Unsupported dask array shape: {dask_array.shape}. Expected 3D or 5D."
            )

        current_shape = (z, y, x)
        # confirm volumes are divisibly sized by the volume size
        pad_widths = []
        for dim_size, vol_size in zip(
            current_shape, (self.volume_size, self.volume_size, self.volume_size)
        ):
            remainder = dim_size % vol_size
            pad = 0 if remainder == 0 else vol_size - remainder
            pad_widths.append((0, pad))  # only pad at the end of each dimension
        full_pad = [
            (0, 0),
            (0, 0),
        ] + pad_widths  # add zero padding for batch and channel dimensions (no padding needed)
        dask_array = da.pad(
            dask_array, pad_width=full_pad, mode="constant", constant_values=0
        )
        return dask_array

    def add_dataset(self, dataset, dataset_name):
        self.datasets[dataset_name] = dataset

    def load_and_pad_zarr(
        self,
        zarr_file,
        scale,
        dataset_name="volume",
        zarr_version="2.0"
    ):
        # unique_name = f"{Path(zarr_file).stem}-{scale}-{uuid.uuid4().hex[:8]}"
        src_file = zarr_file + f"/{scale}/"
        try:
            dask_array = OMEZarrReader(
                data_path=zarr_file,
                multiscale=str(scale),
                zarr_version=zarr_version,
            ).as_dask_array()
        except Exception as e:
            raise ValueError(f"Failed to load zarr file: {src_file}. Error: {e}")

        if self.do_pad_array:
            dask_array = self.pad_dask_array(dask_array)

        if (
            len(dask_array.shape) == 3
        ):  # if the array is 3D, add batch and channel dimensions
            dask_array = da.expand_dims(dask_array, axis=(0, 1))

        xarray_array = xr.DataArray(
            dask_array,
            dims=["T", "C", "Z", "Y", "X"],
            coords={
                "T": np.arange(dask_array.shape[0]),
                "C": np.arange(dask_array.shape[1]),
                "Z": np.arange(dask_array.shape[2]),
                "Y": np.arange(dask_array.shape[3]),
                "X": np.arange(dask_array.shape[4]),
            },
        )
        self.add_dataset(xarray_array, dataset_name)

    def init_batch_generator(self):
        overlap_px = int(self.volume_size * self.overlap_pct)
        self.batch_generator = xbatcher.BatchGenerator(
            xr.Dataset(self.datasets),
            input_dims={
                "T": 1,
                "C": 1,
                "Z": self.volume_size,
                "Y": self.volume_size,
                "X": self.volume_size,
            },
            input_overlap={
                "T": 0,
                "C": 0,
                "Z": overlap_px,
                "Y": overlap_px,
                "X": overlap_px,
            },
        )
        self.indices = [i for i in range(len(self.batch_generator))]

    def _lazy_init_generator(self):
        if self._is_initialized:
            return
        self.init_batch_generator()
        self._is_initialized = True

    @property
    def summary(self):
        return {
            "sample": self.sample,
            "zarr_file": self.zarr_file,
            "scale": self.scale,
            "volume_size": self.volume_size,
            "platform": self.platform,
            "num_volumes": len(self),  # may be filtered in subclasses
            "total_possible": len(self.batch_generator) if self._is_initialized else None,
        }

    def __len__(self):
        if self._is_initialized:
            # Use indices length if present (supports filtering)
            if hasattr(self, "indices"):
                return len(self.indices)
            return len(self.batch_generator)
        else:
            return int(np.prod(self.datasets["volume"].shape) / self.volume_size**3)

    def __getitem__(self, idx):
        self._lazy_init_generator()
        batch = self.batch_generator[int(self.indices[idx])]
        origin = {i: int(batch.coords[i].values[0]) for i in ["T", "C", "Z", "Y", "X"]}
        volume = np.squeeze(batch["volume"].data)[np.newaxis, ...]  # add channel dim back

        if self.transform:
            volume = self.transform({"image": volume})["image"]

        return_dict = {
            "volume": volume,
            "platform": self.platform,
            "sample": self.sample,
            "scale": self.scale,
            "T": origin["T"],
            "C": origin["C"],
            "Z": origin["Z"],
            "Y": origin["Y"],
            "X": origin["X"],
        }

        if "annotations" in batch.data_vars:
            return_dict["annotations"] = batch["annotations"].data

        return volume


class MaskedZarrDataset(ZarrDataset):
    def __init__(
        self,
        *args,
        mask_file,
        downsampled_mask_level=None,
        mask_threshold=0.5,
        force_refilter=False,
        block_prefilter=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.mask_file = mask_file
        self.mask_threshold = mask_threshold
        self.force_refilter = force_refilter
        self.block_prefilter = block_prefilter

        if downsampled_mask_level is None:
            downsampled_mask_level = self.scale + 2  # default offset

        self.downsampled_mask_level = downsampled_mask_level
        # internal flag so we don't double-filter
        self._did_filter = False

        # Load mask at the same scale as the data
        try:
            mask_da = OMEZarrReader(
                data_path=self.mask_file,
                multiscale=str(self.scale),
                zarr_version="2.0",
            ).as_dask_array()
        except Exception as e:
            print(f"\n\tFailed to load mask zarr file: {self.mask_file} - scale {self.scale}. Error: {e}")
            raise
        
        self.original_shape = mask_da.shape
        self.mask_da = self.pad_dask_array(mask_da)

        # Filter valid indices based on mask at initialization
        self._filter_valid_indices()
    
    def _get_cache_path(self):
        """Generate a cache file path for the filtered indices."""
        # Use a local cache directory instead of trying to write to S3
        import hashlib
        from pathlib import Path
        
        # Create cache in user's home directory or temp directory
        cache_base = Path.home() / ".cache" / "aind_octo_data_loaders" / "mask_cache"
        cache_base.mkdir(parents=True, exist_ok=True)
        
        # Create a unique hash for the dataset to handle S3 paths and long names
        dataset_str = f"{self.zarr_file}_{self.mask_file}_scale{self.scale}_vol{self.volume_size}_thr{self.mask_threshold}"
        dataset_hash = hashlib.md5(dataset_str.encode()).hexdigest()
        
        cache_file = cache_base / f"{dataset_hash}.npy"
        return cache_file

    def _filter_valid_indices(self):
        """Filter valid indices from a coarse mask level and upscale to current resolution (vectorized)."""
        if self._did_filter and not self.force_refilter:
            return

        if not self._is_initialized:
            self._lazy_init_generator()

        cache_path = self._get_cache_path()
        if cache_path.exists() and not self.force_refilter:
            try:
                valid_indices = np.load(cache_path).tolist()
                print(f"\n\tLoaded cached filtered indices for {self.sample} at scale {self.scale}: {len(valid_indices)} volumes")
                self.indices = valid_indices
                return
            except Exception as e:
                print(f"\n\tFailed to load cache, recomputing: {e}")

        # Coarse filtering
        coarse_prefilter_level = self.downsampled_mask_level
        print(f"\n\tUsing coarse prefilter from level {coarse_prefilter_level} -> scale {self.scale}")

        try:
            mask_lowres = OMEZarrReader(
                data_path=self.mask_file,
                multiscale=str(coarse_prefilter_level),
                zarr_version="2.0",
            ).as_dask_array()
        except Exception as e:
            print(f"\tCould not load coarse mask: {e}")
            return

        # Going to the coarse block size to keep the same physical space
        downsample_factor = 2 ** (coarse_prefilter_level - self.scale)
        v_coarse = max(1, self.volume_size // downsample_factor)

        # Drop TC in TCZYX in downsampled mask
        mask_lowres = self.pad_dask_array(mask_lowres)[0, 0]
        z_blocks = mask_lowres.shape[0] // v_coarse
        y_blocks = mask_lowres.shape[1] // v_coarse
        x_blocks = mask_lowres.shape[2] // v_coarse

        # Getting coarse blocks that pass the threshold as a grid
        trimmed = mask_lowres[:z_blocks*v_coarse, :y_blocks*v_coarse, :x_blocks*v_coarse]
        block_max = trimmed.reshape(z_blocks, v_coarse, y_blocks, v_coarse, x_blocks, v_coarse).max(axis=(1,3,5)).compute()
        # Condition to make block pass
        coarse_passed = np.argwhere(block_max >= self.mask_threshold)

        if len(coarse_passed) == 0:
            print(f"\tCoarse prefilter found no valid blocks; all blocks assumed empty.")
            self.indices = []
            np.save(cache_path, np.array(self.indices))
            self._did_filter = True
            return

        print(f"\tCoarse prefilter kept {len(coarse_passed)} coarse blocks out of {z_blocks*y_blocks*x_blocks} total.")

        # full-resolution block grid
        mask_core = self.mask_da[0, 0]
        v = self.volume_size
        z_blocks_hr = mask_core.shape[0] // v
        y_blocks_hr = mask_core.shape[1] // v
        x_blocks_hr = mask_core.shape[2] // v

        # vectorized expansion, faster than traditional loops
        # number of full-res blocks per coarse block along each axis
        step = int(np.ceil(v_coarse / v))
        dz, dy, dx = np.meshgrid(np.arange(step), np.arange(step), np.arange(step), indexing="ij")
        dz = dz.ravel()
        dy = dy.ravel()
        dx = dx.ravel()

        # Repeat for all coarse blocks
        zc = np.repeat(coarse_passed[:, 0], len(dz))
        yc = np.repeat(coarse_passed[:, 1], len(dy))
        xc = np.repeat(coarse_passed[:, 2], len(dx))

        # Using xbatcher ordering to compute full-res block indices
        z_hr = zc * step + np.tile(dz, len(coarse_passed))
        y_hr = yc * step + np.tile(dy, len(coarse_passed))
        x_hr = xc * step + np.tile(dx, len(coarse_passed))

        # Keep only blocks within bounds
        mask = (z_hr < z_blocks_hr) & (y_hr < y_blocks_hr) & (x_hr < x_blocks_hr)
        z_hr, y_hr, x_hr = z_hr[mask], y_hr[mask], x_hr[mask]

        # Convert Z,Y,X to linear indices
        valid_indices = (z_hr * y_blocks_hr + y_hr) * x_blocks_hr + x_hr

        print(f"\t{len(valid_indices)}/{len(self.indices)} valid indices after masking.")

        # Save to cache
        np.save(cache_path, np.array(valid_indices))
        self.indices = valid_indices.tolist()
        self._did_filter = True

    def __getitem__(self, idx):
        """Return data without rejection sampling since indices are pre-filtered."""
        return super().__getitem__(idx)
