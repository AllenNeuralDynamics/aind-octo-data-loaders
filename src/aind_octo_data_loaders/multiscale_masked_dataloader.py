"""
Multiscale Masked Zarr DataLoader.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import dask.array as da
import numpy as np
from aind_large_scale_prediction.io import OMEZarrReader

from aind_octo_data_loaders.cloud_mask_dataloader import MaskedZarrDataset
from aind_octo_data_loaders.utils import get_resolution, read_top_level_zattrs


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
        images           : np.ndarray (N_levels, 1, Z, Y, X)  float32
        batch_resolutions: np.ndarray (N_levels, 3)  physical voxel spacing (z,y,x)
        bounding_boxes   : np.ndarray (N_levels, 2, 3)  world-space bbox [[min],[max]]
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

        # Physical voxel resolution (z, y, x) per pyramid level.
        # Read from OME-Zarr coordinateTransformations; fall back to scale
        # factors relative to the base level if metadata is unavailable.
        self.ms_resolutions: Dict[int, Tuple[float, float, float]] = {}
        try:
            zattrs = read_top_level_zattrs(self.zarr_file, anon=False)
            for lvl in self.pyramid_levels:
                res = get_resolution(zattrs, lvl)
                if res is not None:
                    self.ms_resolutions[lvl] = tuple(float(r) for r in res)
        except Exception:
            pass  # fall through to factor-based fallback below

        # This is a fallback in case metadata is missing or malformed.
        # It ensures we always have some resolution values to work with.
        # Ideally, all datasets must pass through the OME-Zarr metadata
        for lvl in self.pyramid_levels:
            if lvl not in self.ms_resolutions:
                factors = self.ms_scale_factors[lvl]
                self.ms_resolutions[lvl] = tuple(float(f) for f in factors)

    def __getitem__(self, idx: int) -> dict:
        # Ensure mask filtering has run (lazy, cached, distributed-safe).
        self._ensure_filtered()

        # Map idx into the pre-filtered xbatcher indices.
        if self.random_indices:
            batch_idx = np.random.randint(0, len(self.batch_generator))
        else:
            batch_idx = int(self.indices[idx % len(self.indices)])

        batch = self.batch_generator[batch_idx]

        v = self.volume_size
        origin = {
            dim: int(batch.coords[dim].values[0]) for dim in ["Z", "Y", "X"]
        }
        base_coords: Tuple[slice, slice, slice] = (
            slice(origin["Z"], origin["Z"] + v),
            slice(origin["Y"], origin["Y"] + v),
            slice(origin["X"], origin["X"] + v),
        )

        level_images = []
        level_resolutions = []
        level_bboxes = []
        level_coords = []

        for lvl in self.pyramid_levels:
            arr = self.ms_lazy_arrays[lvl]
            arr_shape = arr.shape[-3:]
            factors = self.ms_scale_factors[lvl]
            res = self.ms_resolutions[lvl]

            if lvl == self.scale:
                img = np.squeeze(batch["volume"].values)
                lvl_coords = base_coords
            else:
                scaled = _scale_slices(base_coords, factors)
                lvl_coords: Tuple[slice, slice, slice] = tuple(  # type: ignore[assignment]
                    _centered_context_slice((s.start + s.stop) // 2, v, sz)
                    for s, sz in zip(scaled, arr_shape)
                )
                level_slice = _build_level_slice(arr.ndim, lvl_coords)
                img = np.squeeze(arr[level_slice].compute())

            bbox = np.array(
                [
                    [lvl_coords[i].start * res[i] for i in range(3)],
                    [lvl_coords[i].stop * res[i] for i in range(3)],
                ],
                dtype=np.float32,
            )  # (2, 3): [[z_min, y_min, x_min], [z_max, y_max, x_max]]

            level_coords.append(
                np.array(
                    [
                        [lvl_coords[i].start for i in range(3)],
                        [lvl_coords[i].stop for i in range(3)],
                    ],
                    dtype=np.int32,
                )
            )

            level_images.append(
                img[np.newaxis].astype(np.float32)
            )  # (1, Z, Y, X)
            level_resolutions.append(np.array(res, dtype=np.float32))  # (3,)
            level_bboxes.append(bbox)  # (2, 3)

        return {
            "images": np.stack(level_images, axis=0),  # (N_levels, 1, Z, Y, X)
            "batch_resolutions": np.stack(level_resolutions),  # (N_levels, 3)
            "world_coords": np.stack(level_bboxes),  # (N_levels, 2, 3)
            "voxel_coords": np.stack(level_coords),  # (N_levels, 2, 3)
            "sample": self.sample,
            "platform": self.platform,
            "scale": self.scale,
        }
