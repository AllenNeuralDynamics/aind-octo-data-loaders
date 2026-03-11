"""
Multiscale Masked Zarr DataLoader.

Combines:
- MaskedZarrDataset  (cloud-mask filtering, caching, lazy init)
- Multiscale context loading  (same pixel-size patches at every pyramid level,
  high-res patch centred inside downsampled context patches)
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import dask.array as da
import numpy as np
from aind_large_scale_prediction.io import OMEZarrReader

from aind_octo_data_loaders.cloud_mask_dataloader import MaskedZarrDataset


def _compute_scale_factors(
    base_shape: Sequence[int],
    target_shape: Sequence[int],
    max_rel_error: float = 0.01,
) -> Tuple[int, ...]:
    scale = []
    for base, target in zip(base_shape, target_shape):
        factor = base / target
        nearest = round(factor)
        if nearest == 0:
            raise ValueError("Invalid scale factor computed as 0.")
        if abs(factor - nearest) / nearest > max_rel_error:
            raise ValueError(
                f"Non-integer scale factor: base={base}, target={target}"
            )
        scale.append(int(nearest))
    return tuple(scale)


def _downscale_slice(s: slice, factor: int) -> slice:
    return slice(int(s.start / factor), int(s.stop / factor))


def _scale_slices(
    slices: Sequence[slice], factors: Sequence[int]
) -> Tuple[slice, ...]:
    return tuple(_downscale_slice(s, f) for s, f in zip(slices, factors))


def _centered_context_slice(center: int, target: int, arr_size: int) -> slice:
    """Slice of length `target` centred on `center`, clamped to [0, arr_size)."""
    start = max(0, center - target // 2)
    end = start + target
    if end > arr_size:
        end = arr_size
        start = max(0, arr_size - target)
    return slice(start, end)


def _build_level_slice(ndim: int, coords: Tuple[slice, slice, slice]) -> tuple:
    if ndim == 3:
        return coords
    if ndim == 4:
        return (slice(None),) + coords
    raise ValueError(f"Unsupported ndim={ndim}. Expected 3 or 4.")


class MultiScaleMaskedDataset(MaskedZarrDataset):
    """
    Extends MaskedZarrDataset to return same-voxel-size patches across
    multiple pyramid levels. The pyramid levels load the chunk assuming
    the center of the highest resolution volume as origin.

    Parameters
    ----------
    pyramid_levels : sequence of int
        Pyramid levels to load.  Must include `scale` (the base level).
        Lower numbers = higher resolution (standard OME-Zarr convention).
    All other parameters are forwarded to MaskedZarrDataset.

    Returns
    -------
    dict with keys:
        images      : dict[level -> np.ndarray]  shape (Z, Y, X) for every level
        coords      : dict[level -> (z_slice, y_slice, x_slice)]
                      absolute pixel slices in each level's coordinate space
        base_coords : (z_slice, y_slice, x_slice) in base-level pixel space
        sample, platform, scale : forwarded from parent
    """

    def __init__(
        self,
        *args,
        pyramid_levels: Sequence[int],
        random_indices: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.pyramid_levels = sorted(pyramid_levels)
        self.random_indices = random_indices

        if self.scale not in self.pyramid_levels:
            raise ValueError(
                f"Base scale={self.scale} must be included in pyramid_levels={pyramid_levels}."
            )

        # Load a lazy dask array for every pyramid level.
        # The base level is already in self.datasets["volume"] (xarray), but we
        # also keep a raw dask array here for shape queries.
        self.ms_lazy_arrays: Dict[int, da.Array] = {}
        for lvl in self.pyramid_levels:
            reader = OMEZarrReader(
                data_path=self.zarr_file,
                multiscale=str(lvl),
                zarr_version=self.zarr_version,
            )
            self.ms_lazy_arrays[lvl] = da.squeeze(reader.as_dask_array())

        # Scale factors: how many base-level pixels per downsampled pixel.
        base_shape = self.ms_lazy_arrays[self.scale].shape[-3:]
        self.ms_scale_factors: Dict[int, Tuple[int, ...]] = {
            lvl: _compute_scale_factors(
                base_shape, self.ms_lazy_arrays[lvl].shape[-3:]
            )
            for lvl in self.pyramid_levels
        }

    def __getitem__(self, idx: int) -> dict:
        # Ensure mask filtering has run (lazy, cached, distributed-safe).
        self._ensure_filtered()

        # Map idx into the pre-filtered xbatcher indices.
        if self.random_indices:
            batch_idx = np.random.randint(0, len(self.batch_generator))
        else:
            batch_idx = int(self.indices[idx % len(self.indices)])

        batch = self.batch_generator[batch_idx]

        # ZarrDataset builds the DataArray with coords {"Z": np.arange(shape[2]), ...}
        # so batch.coords["Z"].values[0] is the absolute start pixel in the
        # padded base-level array.
        v = self.volume_size
        origin = {
            dim: int(batch.coords[dim].values[0]) for dim in ["Z", "Y", "X"]
        }
        base_coords: Tuple[slice, slice, slice] = (
            slice(origin["Z"], origin["Z"] + v),
            slice(origin["Y"], origin["Y"] + v),
            slice(origin["X"], origin["X"] + v),
        )

        images: Dict[int, np.ndarray] = {}
        coords: Dict[int, Tuple[slice, slice, slice]] = {}

        for lvl in self.pyramid_levels:
            arr = self.ms_lazy_arrays[lvl]
            arr_shape = arr.shape[-3:]
            factors = self.ms_scale_factors[lvl]

            if lvl == self.scale:
                # Base level: xbatcher already loaded this data.
                images[lvl] = np.squeeze(batch["volume"].values)
                coords[lvl] = base_coords

            else:
                # Downsampled level: derive where the HR patch falls in this
                # level's pixel space, then build a same-size context window.
                scaled = _scale_slices(base_coords, factors)

                ctx: Tuple[slice, slice, slice] = tuple(  # type: ignore[assignment]
                    _centered_context_slice((s.start + s.stop) // 2, v, sz)
                    for s, sz in zip(scaled, arr_shape)
                )
                coords[lvl] = ctx

                level_slice = _build_level_slice(arr.ndim, ctx)
                images[lvl] = np.squeeze(arr[level_slice].compute())

        return {
            "images": images,
            "coords": coords,
            "base_coords": base_coords,
            "sample": self.sample,
            "platform": self.platform,
            "scale": self.scale,
        }
