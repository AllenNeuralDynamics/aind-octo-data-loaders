"""
Multiscale masked Zarr dataloader.
"""

from __future__ import annotations

import inspect
import warnings
from typing import Dict, List, Optional, Sequence, Tuple

import dask.array as da
import numpy as np
from aind_large_scale_prediction.io import OMEZarrReader

from aind_octo_data_loaders.cloud_mask_dataloader import (
    MaskedZarrDataset,
    ZarrDataset,
)
from aind_octo_data_loaders.utils import get_resolution, read_top_level_zattrs


def _compute_scale_factors(
    base_shape: Sequence[int],
    target_shape: Sequence[int],
    max_rel_error: float = 0.01,
) -> Tuple[int, ...]:
    """
    Computes the scale factors between the base shape
    and the target shape.

    Parameters
    ----------
    base_shape: Sequence[int]
        Shape of the base array.
    target_shape: Sequence[int]
        Shape of the target array.
    max_rel_error: float
        Maximum relative error allowed when computing the scale factor.

    Returns
    -------
    Tuple[int, ...]
        Scale factors for each dimension.

    """
    # Array for the new scale
    scale = []
    for base, target in zip(base_shape, target_shape):
        factor = base / target

        # Rounding to the nearest integer
        nearest = round(factor)
        if nearest == 0:
            raise ValueError("Invalid scale factor computed as 0.")
        if abs(factor - nearest) / nearest > max_rel_error:
            raise ValueError(f"Non-integer scale factor: base={base}, target={target}")
        scale.append(int(nearest))
    return tuple(scale)


def _downscale_slice(s: slice, factor: int) -> slice:
    """
    Downscale a slice by a given factor.

    Parameters
    ----------
    s : slice
        Original slice.
    factor : int
        Downscaling factor.

    Returns
    -------
    slice
        Downscaled slice.
    """
    return slice(int(s.start / factor), int(s.stop / factor))


def _scale_slices(slices: Sequence[slice], factors: Sequence[int]) -> Tuple[slice, ...]:
    """
    Scale a sequence of slices by corresponding factors.

    Parameters
    ----------
    slices : Sequence[slice]
        Original slices.
    factors : Sequence[int]
        Scaling factors for each dimension.

    Returns
    -------
    Tuple[slice, ...]
        Scaled slices.
    """
    return tuple(_downscale_slice(s, f) for s, f in zip(slices, factors))


def _centered_context_slice(center: int, target: int, arr_size: int) -> slice:
    """
    Given a center index, a target size, and the size of the array,
    returns slices that are centered around the given center.

    Parameters
    ----------
    center: int
        Center index around which to create the slice.
    target: int
        Target size of the slice.
    arr_size: int
        Size of the array along the dimension of interest.

    Returns
    -------
    slice
        Centered slice.
    """
    start = max(0, center - target // 2)
    # Moving to the left most end and then adding
    # the final target
    end = start + target
    if end > arr_size:
        end = arr_size
        start = max(0, arr_size - target)
    return slice(start, end)


def _centered_context_slice_with_flag(
    center: int, target: int, arr_size: int
) -> Tuple[slice, bool]:
    """
    Return centered slice and whether no clamping was needed.
    Parameters
    ----------
    center: int
        Center index around which to create the slice.
    target: int
        Target size of the slice.
    arr_size: int
        Size of the array along the dimension of interest.

    Returns
    -------
    Tuple[slice, bool]
        Centered slice and a boolean indicating whether no clamping was needed.

    """
    ideal_start = center - target // 2
    ideal_end = ideal_start + target
    out = _centered_context_slice(center=center, target=target, arr_size=arr_size)
    exact = (out.start == ideal_start) and (out.stop == ideal_end)
    return out, exact


def _build_level_slice(ndim: int, coords: Tuple[slice, slice, slice]) -> tuple:
    """
    Builds level slice for the given number of dimensions.

    Parameters
    ----------
    ndim : int
        Number of dimensions of the array.
    coords : Tuple[slice, slice, slice]
        Slices for the last three dimensions.

    Returns
    -------
    tuple
        Slices for the array with the given number of dimensions.

    """
    curr_coords_len = len(coords)
    ndim = int(ndim)

    add_dims = ndim - curr_coords_len

    if add_dims < 0:
        raise ValueError(f"Cannot build level slice: ndim={ndim} < len(coords)={curr_coords_len}")

    for i in range(add_dims):
        coords = (slice(None),) + coords

    return coords


def _normalize_resolution(
    res: Sequence[float],
    resolution_axis_order: str,
    input_resolution_order: Optional[str] = "zyx",
) -> Tuple[float, float, float]:
    """
    Normalizes the resolution to a given order.

    Parameters
    ----------
    res: Sequence[float]
        Resolution values for the three dimensions.
    resolution_axis_order: str
        Axis order of the resolution. Can be any combinaton of
        the characters 'z', 'y', and 'x'.
    input_resolution_order: Optional[str]
        Axis order of the input resolution. Can be any combination of
        the characters 'z', 'y', and 'x'. Default is 'zyx'.

    Returns
    -------
    Tuple[float, float, float]
        Normalized resolution in 'zyx' order.

    """
    if len(res) != 3:
        raise ValueError(f"Resolution must have 3 values, got {len(res)}")

    if len(res) != len(input_resolution_order):
        raise ValueError(
            f"Resolution length {len(res)} does not match input_resolution_order length {len(input_resolution_order)}"
        )

    input_resolution_order = input_resolution_order.lower()
    resolution_axis_order = resolution_axis_order.lower()

    hashmap = {input_resolution_order[i]: res[i] for i in range(len(res))}

    values = tuple(float(v) for v in res)
    if any(v <= 0 for v in values):
        raise ValueError(f"Resolution must be positive, got {values}")

    ordered_resolution = (float(hashmap[axis]) for axis in resolution_axis_order)

    return ordered_resolution


class _MultiScaleViTCore:
    """Shared geometry logic for masked and unmasked datasets."""

    def _init_state(
        self,
        pyramid_levels: Sequence[int],
        max_resample_tries: int,
        resolution_axis_order: str,
        enforce_exact_center: Optional[bool] = False,
        random_indices: Optional[bool] = False,
    ) -> None:
        """
        Initiates the state of the multi-scale ViT core.

        Parameters
        ----------
        pyramid_levels : Sequence[int]
            Sequence of pyramid levels as integers that will
            be used in the data loading.

        enforce_exact_center : Optional[bool] = False
            If True, the center of the patch will be exactly at the center of the volume.
            If False, the center may be slightly off. Default is False.

        max_resample_tries : int
            Maximum number of times to resample a patch if the center is not valid.

        resolution_axis_order : str
            Axis order of the resolution. Can be any combinaton of
            the characters 'z', 'y', and 'x'.

        random_indices : Optional[bool] = False
            If True, indices will be sampled randomly. If False,
            indices will be sampled sequentially. Default is False.
        """
        self.pyramid_levels = sorted(pyramid_levels)
        self.random_indices = random_indices
        self.enforce_exact_center = enforce_exact_center
        self.max_resample_tries = int(max_resample_tries)
        self.resolution_axis_order = resolution_axis_order
        self.fallback_resolution = (1.0, 1.0, 1.0)

        if self.scale not in self.pyramid_levels:
            raise ValueError(
                f"Base scale={self.scale} must be included in pyramid_levels={pyramid_levels}."
            )

        # Loads the multi-scale lazy arrays
        self.ms_lazy_arrays: Dict[int, da.Array] = {}
        for lvl in self.pyramid_levels:
            reader = OMEZarrReader(
                data_path=self.zarr_file,
                multiscale=str(lvl),
                zarr_version=self.zarr_version,
            )
            self.ms_lazy_arrays[lvl] = da.squeeze(reader.as_dask_array())

        # Base array shape of the highest resolution scale
        # This identifies the scale factor based on the pyramid level shapes
        base_shape = self.ms_lazy_arrays[self.scale].shape[-3:]
        self.ms_scale_factors: Dict[int, Tuple[int, ...]] = {
            lvl: _compute_scale_factors(base_shape, self.ms_lazy_arrays[lvl].shape[-3:])
            for lvl in self.pyramid_levels
        }

        # Fetching the resolutions from the metadata
        self.ms_resolutions: Dict[int, Tuple[float, float, float]] = {}
        try:
            zattrs = read_top_level_zattrs(self.zarr_file, anon=False)
            for lvl in self.pyramid_levels:
                res = get_resolution(zattrs, lvl)
                if res is not None:
                    self.ms_resolutions[lvl] = _normalize_resolution(
                        res, self.resolution_axis_order
                    )
                else:
                    # Send warning if resolution is not found
                    raise RuntimeError(
                        f"Resolution not found for pyramid level {lvl}. "
                        f"Using default resolution: {self.fallback_resolution}"
                    )
        except Exception as e:
            raise RuntimeError(
                "Failed to read top-level Zarr attributes. "
                "Using default resolutions for all pyramid levels. "
                f"Error: {e}"
            )

        # Fallback resolution but I deactivated it by adding the runtime error above
        base_res = self.ms_resolutions.get(self.scale, self.fallback_resolution)

        # Downsampled resolutions with the scale factors
        for lvl in self.pyramid_levels:
            if lvl not in self.ms_resolutions:
                fz, fy, fx = self.ms_scale_factors[lvl]
                self.ms_resolutions[lvl] = (
                    float(base_res[0] * fz),
                    float(base_res[1] * fy),
                    float(base_res[2] * fx),
                )

        self.base_resolution = self.ms_resolutions[self.scale]

    def _prepare_indices(self) -> None:
        """Hook for subclasses to ensure index state is ready."""

    def _sample_batch_idx(self, idx: int) -> int:
        """Sample from available indices in deterministic or random mode."""
        if len(self.indices) == 0:
            raise RuntimeError("No indices available in dataset.")

        if self.random_indices:
            pos = int(np.random.randint(0, len(self.indices)))
            return int(self.indices[pos])

        return int(self.indices[idx % len(self.indices)])

    def _build_level_data(
        self,
        base_coords: Tuple[slice, slice, slice],
        lvl: int,
        base_batch_img: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[slice, slice, slice], bool, np.ndarray]:
        """
        Builds
        """
        v = self.volume_size
        arr = self.ms_lazy_arrays[lvl]
        arr_shape = arr.shape[-3:]
        factors = self.ms_scale_factors[lvl]
        padding_mask = np.zeros((v, v, v), dtype=bool)

        # Nothing to do if it's the highest resolution scale
        if lvl == self.scale:
            return np.squeeze(base_batch_img), base_coords, True, padding_mask

        # Scale the base coordinates of that resolution
        # to the precomputed factors, factors are numbers with
        # the downsampling needed to go from highest res to
        # that pyramid level per axis.
        scaled = _scale_slices(base_coords, factors)

        # moves the coordinates to the exact center from the highest
        coords_with_exact = [
            _centered_context_slice_with_flag(
                center=(s.start + s.stop) // 2,
                target=v,
                arr_size=sz,
            )
            for s, sz in zip(scaled, arr_shape)
        ]
        lvl_coords = tuple(item[0] for item in coords_with_exact)
        is_exact = all(item[1] for item in coords_with_exact)

        # Builds the slice tuple to extract the data from the array
        level_slice = _build_level_slice(arr.ndim, lvl_coords)
        img = np.squeeze(arr[level_slice].compute())

        # Pad to volume_size when the array boundary was hit (arr_size < volume_size).
        # Extend lvl_coords so bbox/world-coords cover the full requested region,
        # and mark padded voxels in padding_mask so consumers can exclude them.
        if img.shape != (v, v, v):
            actual_shape = img.shape
            lvl_coords = tuple(slice(s.start, s.start + v) for s in lvl_coords)
            pad_width = [(0, v - actual_shape[i]) for i in range(3)]
            img = np.pad(img, pad_width, mode="constant", constant_values=0)
            for dim, (_, pad_after) in enumerate(pad_width):
                if pad_after > 0:
                    idx: List = [slice(None), slice(None), slice(None)]
                    idx[dim] = slice(actual_shape[dim], v)
                    padding_mask[tuple(idx)] = True

        return img, lvl_coords, is_exact, padding_mask

    def __getitem__(self, idx: int) -> dict:
        """
        Get item method

        Parameters
        ----------
        idx: int
            Index to return from the dataset
        """
        self._prepare_indices()

        selected_batch = None
        selected_base_coords = None
        selected_level_data = None

        tries = max(1, self.max_resample_tries)
        for attempt in range(tries):
            batch_idx = self._sample_batch_idx(idx if attempt == 0 else idx + attempt)
            batch = self.batch_generator[batch_idx]

            v = self.volume_size
            origin = {dim: int(batch.coords[dim].values[0]) for dim in ["Z", "Y", "X"]}
            base_coords: Tuple[slice, slice, slice] = (
                slice(origin["Z"], origin["Z"] + v),
                slice(origin["Y"], origin["Y"] + v),
                slice(origin["X"], origin["X"] + v),
            )

            base_batch_img = batch["volume"].values

            # Gets data from all pyramid levels
            level_data = []
            all_exact = True
            for lvl in self.pyramid_levels:
                img, lvl_coords, exact, padding_mask = self._build_level_data(
                    base_coords=base_coords,
                    lvl=lvl,
                    base_batch_img=base_batch_img,
                )
                level_data.append((lvl, img, lvl_coords, padding_mask))
                all_exact = all_exact and exact

            selected_batch = batch
            selected_base_coords = base_coords
            selected_level_data = level_data

            if (not self.enforce_exact_center) or all_exact:
                break

        if selected_level_data is None or selected_batch is None or selected_base_coords is None:
            raise RuntimeError("Failed to sample a valid multiscale batch.")

        # Preparing data objects to return
        level_images: List[np.ndarray] = []
        level_resolutions: List[np.ndarray] = []
        level_bboxes: List[np.ndarray] = []
        level_bboxes_fine_px: List[np.ndarray] = []
        level_coords: List[np.ndarray] = []
        level_factors: List[int] = []
        level_factors_zyx: List[np.ndarray] = []
        level_padding_masks: List[np.ndarray] = []

        # Going through selected level datasets
        # at this point, the data is centered across resolutions
        for lvl, img, lvl_coords, padding_mask in selected_level_data:
            res = self.ms_resolutions[lvl]

            # This bounding box is in physical space not in coordinate space!
            bbox = np.array(
                [
                    [lvl_coords[i].start * res[i] for i in range(3)],
                    [lvl_coords[i].stop * res[i] for i in range(3)],
                ],
                dtype=np.float32,
            )

            # convention: bbox in finest-resolution voxel coordinate system.
            # This is useful so RoPE could map resolutions to the same positional curve
            bbox_fine_px = np.array(
                [
                    [bbox[0, i] / self.base_resolution[i] for i in range(3)],
                    [bbox[1, i] / self.base_resolution[i] for i in range(3)],
                ],
                dtype=np.float32,
            )

            # This level coords is in coordinate space
            level_coords.append(
                np.array(
                    [
                        [lvl_coords[i].start for i in range(3)],
                        [lvl_coords[i].stop for i in range(3)],
                    ],
                    dtype=np.int32,
                )
            )

            level_images.append(img.astype(np.float32))
            level_resolutions.append(np.array(res, dtype=np.float32))
            level_bboxes.append(bbox)
            level_bboxes_fine_px.append(bbox_fine_px)
            level_factors_zyx.append(np.array(self.ms_scale_factors[lvl], dtype=np.int32))
            level_factors.append(int(self.ms_scale_factors[lvl][0]))
            level_padding_masks.append(padding_mask)

        level_images_arr = np.stack(level_images, axis=0)
        if self.transform:
            level_images_arr = self.transform({"image": level_images_arr})["image"]
            # Reset padded voxels to zero after normalization so they carry no
            # signal — normalization would otherwise shift zeros to (0-mean)/std.
            pad_arr = np.stack(level_padding_masks, axis=0)  # (N_levels, Z, Y, X)
            level_images_arr[pad_arr] = 0.0

        level_images_arr = level_images_arr[:, np.newaxis, ...]

        return {
            "images": level_images_arr,
            "batch_resolutions": np.stack(level_resolutions),
            "world_coords": np.stack(level_bboxes),
            "world_coords_fine_vx": np.stack(level_bboxes_fine_px),
            "levels": np.array(level_factors, dtype=np.int32),
            "levels_zyx": np.stack(level_factors_zyx),
            "voxel_coords": np.stack(level_coords),
            "padding_mask": np.stack(level_padding_masks),
            "sample": self.sample,
            "platform": self.platform,
        }


class MultiScaleMaskedDatasetViT(_MultiScaleViTCore, MaskedZarrDataset):
    """
    Multiscale masked dataset

    Returns
    -------
    dict with keys:
        images           : np.ndarray (N_levels, 1, Z, Y, X) float32
        batch_resolutions: np.ndarray (N_levels, 3) in (z, y, x)
        world_coords     : np.ndarray (N_levels, 2, 3) [[min],[max]] in world units
        bbox_fine_px     : np.ndarray (N_levels, 2, 3) bbox in finest-level pixel coordinates
        levels           : np.ndarray (N_levels,) legacy scalar level factors
        levels_zyx       : np.ndarray (N_levels, 3) anisotropic downsampling factors
        voxel_coords     : np.ndarray (N_levels, 2, 3)
        sample, platform, scale
    """

    def __init__(
        self,
        *args,
        pyramid_levels: Sequence[int],
        random_indices: bool = False,
        enforce_exact_center: bool = True,
        max_resample_tries: int = 32,
        resolution_axis_order: str = "zyx",
        **kwargs,
    ) -> None:
        """
        Check _MultiscaleViTCore parameters
        """
        super().__init__(*args, **kwargs)
        self._init_state(
            pyramid_levels=pyramid_levels,
            random_indices=random_indices,
            enforce_exact_center=enforce_exact_center,
            max_resample_tries=max_resample_tries,
            resolution_axis_order=resolution_axis_order,
        )

    def _prepare_indices(self) -> None:
        """
        Prepares indices
        """
        self._ensure_filtered()


class MultiScaleDatasetViT(_MultiScaleViTCore, ZarrDataset):
    """
    Multiscale dataset ViT without mask support
    """

    def __init__(
        self,
        *args,
        pyramid_levels: Sequence[int],
        random_indices: bool = False,
        enforce_exact_center: bool = True,
        max_resample_tries: int = 32,
        resolution_axis_order: str = "zyx",
        **kwargs,
    ) -> None:
        """
        Check _MultiscaleViTCore parameters
        """
        super().__init__(*args, **kwargs)

        self._init_state(
            pyramid_levels=pyramid_levels,
            random_indices=random_indices,
            enforce_exact_center=enforce_exact_center,
            max_resample_tries=max_resample_tries,
            resolution_axis_order=resolution_axis_order,
        )

    def _prepare_indices(self) -> None:
        """
        Prepares indices
        """
        # Ensure xbatcher is initialized and index list is available.
        self._lazy_init_generator()


def remove_class_params(cls, kwargs):
    """
    Utility function to remove class parameters
    """
    # Get the __init__ signature
    sig = inspect.signature(cls.__init__)

    # Get parameter names (skip 'self')
    param_names = set(sig.parameters.keys()) - {"self"}

    # Remove those keys from kwargs
    for k, v in kwargs.items():
        if k not in param_names:
            filtered_kwargs = {k: v}
        else:
            new_kwargs = {k: v}

    return new_kwargs, filtered_kwargs


def build_multires_dataset(*, mask_file=None, **kwargs):
    """
    Build masked or unmasked multiresolution dataset based on mask availability.
    """
    if (mask_file is None) or (isinstance(mask_file, str) and mask_file.strip() == ""):
        # Removes parameters that are not present as in kwargs
        kwargs, filtered = remove_class_params(cls=MultiScaleDatasetViT, kwargs=kwargs)
        warnings.warn(f"MultiScaleDatasetViT does not accept these parameters: {filtered}")
        return MultiScaleDatasetViT(**kwargs)

    return MultiScaleMaskedDatasetViT(mask_file=mask_file, **kwargs)
