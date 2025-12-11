"""
Concatenated zarr iterable dataset
"""

from typing import Callable, List, Optional, Union, Dict, Literal

from torch.utils.data import ChainDataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from zarrdataset import (
    ImagesDatasetSpecs,
    MasksDatasetSpecs,
    PatchSampler,
    BlueNoisePatchSampler,
    ZarrDataset,
    zarrdataset_worker_init_fn,
    chained_zarrdataset_worker_init_fn,
)
from zarrdataset._zarrdataset import ImageSample, get_ddp_info
import random
import numpy as np
from .utils import extract_wavelengths, get_resolution, read_top_level_zattrs

class CustomZarrDataset(ZarrDataset):
    def __init__(
        self,
        *args,
        wavelength_nm=None,
        numerical_aperture=None,
        image_resolution=None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.wavelength_nm = wavelength_nm
        self.numerical_aperture = numerical_aperture
        self.image_resolution = image_resolution
    
    def __iter__(self):
        # Preload the files and masks associated with them
        self._initialize()

        samples = [
            ImageSample(im_id, chk_id, shuffle=self._shuffle)
            for im_id in range(len(self._arr_lists))
            for chk_id in range(len(self._toplefts[im_id]))
        ]

        # Add sharding here
        rank, world_size = get_ddp_info()
        samples = [s for i, s in enumerate(samples) if i % world_size == rank]

        # Shuffle chunks here if samples will come from the same chunk until
        # they are depleted.
        if self._shuffle and self._draw_same_chunk:
            random.shuffle(samples)

        prev_im_id = -1
        prev_chk_id = -1
        prev_chk = -1
        curr_chk = 0
        self._curr_collection = None

        while samples:
            # Shuffle chunks here if samples can come from different chunks.
            if self._shuffle and not self._draw_same_chunk:
                curr_chk = random.randrange(0, len(samples))

            im_id = samples[curr_chk].im_id
            chk_id = samples[curr_chk].chk_id

            chunk_tlbr = self._toplefts[im_id][chk_id]

            # If this sample is from a different image or chunk, free the
            # previous sample and re-sample the patches from the current chunk.
            if prev_im_id != im_id or chk_id != prev_chk_id:
                if prev_chk >= 0:
                    # Free the patch ordering from the previous chunk to save
                    # memory.
                    samples[prev_chk].free_sampler()

                prev_chk = curr_chk
                prev_chk_id = chk_id

                if prev_im_id != im_id:
                    prev_im_id = im_id
                    self._curr_collection = self._arr_lists[im_id]

                if self._patch_sampler is not None:
                    patches_tls = self._patch_sampler.compute_patches(
                        self._curr_collection,
                        chunk_tlbr
                    )

                else:
                    patches_tls = [chunk_tlbr]

                samples[curr_chk].num_patches = len(patches_tls)

                if not len(patches_tls):
                    samples.pop(curr_chk)
                    prev_chk = -1
                    continue

            # # Initialize the count of top-left positions for patches inside
            # # this chunk.
            curr_patch, is_empty = samples[curr_chk].next_patch()

            # When all possible patches have been extracted from the current
            # chunk, remove that chunk from the list of samples.
            if is_empty:
                samples.pop(curr_chk)
                prev_chk = -1

            patch_tlbr = patches_tls[curr_patch]
            itemized = self.__getitem__(patch_tlbr)
            patches = itemized['data']

            if self._return_positions:
                pos = [
                    [patch_tlbr[ax].start
                     if patch_tlbr[ax].start is not None else 0,
                     patch_tlbr[ax].stop
                     if patch_tlbr[ax].stop is not None else -1
                     ] if ax in patch_tlbr else [0, -1]
                    for ax in self._collections[self._ref_mod][0]["axes"]
                ]
                patches = [np.array(pos, dtype=np.int64)] + patches

            if self._return_worker_id:
                wid = [np.array(self._worker_id, dtype=np.int64)]
                patches = wid + patches

            if len(patches) > 1:
                patches = tuple(patches)
            else:
                patches = patches[0]

            itemized['data'] = patches
            yield itemized
    
    def __getitem__(self, idx):
        sample = super().__getitem__(idx)
        return {
            'data': sample,
            'wavelength_nm': self.wavelength_nm,
            'numerical_aperture': self.numerical_aperture,
            'image_resolution': self.image_resolution,
        }


class ZarrDatasets:
    """
    A wrapper class to manage multiple Zarr datasets for efficient loading and processing.

    This class handles the creation and management of multiple ZarrDataset instances,
    combining them into a single ChainDataset and providing a DataLoader for batch processing.

    Parameters
    ----------
    dataset_paths : List[str]
        Paths to Zarr dataset files or directories.
    dataset_scales : List[Union[str, int]]
        Scale identifiers for each dataset (e.g., "s0" or 0).
    patch_size_zyx : List[int]
        Dimensions of the patches to extract in Z, Y, X order.
    batch_size : int
        Number of samples per batch.
    axes : str, optional
        String describing the dimension ordering of the data (default: 'TCZYX').
    shuffle : bool, optional
        Whether to shuffle the data during loading (default: True).
    transform : Callable, optional
        Transformation function to apply to each sample (default: None).
    num_workers : int, optional
        Number of worker processes for data loading (default: 4).

    Attributes
    ----------
    zarr_datasets : ChainDataset
        Combined dataset containing all individual ZarrDataset instances.
    individual_datasets : List[ZarrDataset]
        List of individual ZarrDataset instances.
    dataloader : DataLoader
        PyTorch DataLoader for batch loading of data.
    """

    def __init__(
        self,
        dataset_paths: List[str],
        dataset_scales: List[Union[str, int]],
        patch_size_zyx: List[int],
        batch_size: int,
        axes: str = "TCZYX",
        shuffle: bool = True,
        transform: Optional[Callable] = None,
        num_workers: int = 4,
        return_positions: bool = False,
        return_worker_id: bool = False,
        return_dataset_paths: bool = False,
        return_dataset_scales: bool = False,
        sampler_type: Literal["patch","bluenoise"] = "patch",
        **kwargs: Dict[str, Union[str, int, bool, Callable]]
    ):
        self.dataset_paths = dataset_paths
        self.axes = axes
        self.dataset_scales = dataset_scales
        self.patch_size_zyx = patch_size_zyx
        self.transform = transform
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.return_positions = return_positions
        self.return_worker_id = return_worker_id
        self.return_dataset_paths = return_dataset_paths
        self.return_dataset_scales = return_dataset_scales
        self.sampler_type = sampler_type

        # Ensure dataset_paths and dataset_scales have the same length
        if len(dataset_paths) != len(dataset_scales):
            raise ValueError(
                f"Number of dataset paths ({len(dataset_paths)}) must match "
                f"number of dataset scales ({len(dataset_scales)})"
            )

        self.individual_datasets = []
        self._initialize_datasets()

    def _initialize_datasets(self, **kwargs):
        """Initialize the patch sampler and create the datasets."""
        if self.sampler_type == "patch":
            self.sampler = self._create_patch_sampler()
        elif self.sampler_type == "bluenoise":
            self.sampler = self._create_blue_noise_patch_sampler()
        else:
            raise ValueError(
                f"Invalid sampler type: {self.sampler_type}. "
                f"Choose either 'patch' or 'bluenoise'."
            )
        self.individual_datasets = self._create_datasets()
        self.zarr_datasets = ChainDataset(self.individual_datasets)
        self.dataloader = self._create_dataloader(**kwargs)
        # self.individual_dataloaders = self._create_individual_dataloaders()

    def _create_patch_sampler(self) -> PatchSampler:
        """
        Create a patch sampler with the specified patch dimensions.

        Returns
        -------
        PatchSampler
            Configured patch sampler object.

        Raises
        ------
        ValueError
            If patch_size_zyx does not have exactly 3 dimensions.
        """
        if len(self.patch_size_zyx) != 3:
            raise ValueError(
                f"Please provide ZYX patches with exactly 3 dimensions. "
                f"Got {self.patch_size_zyx} with {len(self.patch_size_zyx)} dimensions."
            )

        return PatchSampler(
            dict(
                Z=self.patch_size_zyx[0],
                Y=self.patch_size_zyx[1],
                X=self.patch_size_zyx[2],
            )
        )
    
    def _create_blue_noise_patch_sampler(self) -> BlueNoisePatchSampler:
        """
        Create a blue noise patch sampler with the specified patch dimensions.

        Returns
        -------
        BlueNoisePatchSampler
            Configured patch sampler object.

        Raises
        ------
        ValueError
            If patch_size_zyx does not have exactly 3 dimensions.
        """
        if len(self.patch_size_zyx) != 3:
            raise ValueError(
                f"Please provide ZYX patches with exactly 3 dimensions. "
                f"Got {self.patch_size_zyx} with {len(self.patch_size_zyx)} dimensions."
            )

        return BlueNoisePatchSampler(
            patch_size = dict(
                Z=self.patch_size_zyx[0],
                Y=self.patch_size_zyx[1],
                X=self.patch_size_zyx[2],
            ),
            resample_positions = False,
            allow_overlap = True,
        )

    def _create_datasets(self) -> List[ZarrDataset]:
        """
        Create individual ZarrDataset instances for each input path.

        Returns
        -------
        List[ZarrDataset]
            List of initialized ZarrDataset objects.
        """
        zarr_datasets = []
        for i, (dataset_path, dataset_mask) in enumerate(self.dataset_paths):
            
            # Validate dataset path
            # if not os.path.exists(dataset_path):
            #     raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
            
            # ex, em = extract_wavelengths(dataset_path)
            # zattrs = read_top_level_zattrs(dataset_path, anon=True)
            # resolution = get_resolution(zattrs, self.dataset_scales[i])

            try:
                dataset_specs = [
                    ImagesDatasetSpecs(
                        filenames=[dataset_path],
                        modality="images",
                        source_axes=self.axes,
                        data_group=str(self.dataset_scales[i]),
                        transform=None,#self.transform,
                    )
                ]

                if dataset_mask:
                    dataset_specs.append(
                        MasksDatasetSpecs(
                            filenames=[dataset_mask],
                            modality="masks",
                            source_axes=self.axes,
                            data_group=str(self.dataset_scales[i]),
                            # transform=self.transform,
                        )
                    )

                zarr_datasets.append(
                    ZarrDataset(
                        dataset_specs=dataset_specs,
                        patch_sampler=self.sampler,
                        shuffle=self.shuffle,
                        return_positions=self.return_positions,
                        return_worker_id=self.return_worker_id,
                    )
                )
            
            except Exception as e:
                print(f"Error loading {dataset_path} at scale {self.dataset_scales[i]}. Error: {e}")
            # CustomZarrDataset(
            #     dataset_specs=dataset_specs,
            #     patch_sampler=self.sampler,
            #     shuffle=self.shuffle,
            #     return_positions=self.return_positions,
            #     return_worker_id=self.return_worker_id,
            #     wavelength_nm=ex,
            #     numerical_aperture=1.4,
            #     image_resolution=resolution,
            # )

        return zarr_datasets

    def _create_dataloader(self, **kwargs) -> DataLoader:
        """
        Create a PyTorch DataLoader for the combined datasets.

        Returns
        -------
        DataLoader
            Configured DataLoader for batch processing.
        """
        return DataLoader(
            self.zarr_datasets,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=chained_zarrdataset_worker_init_fn,
            **kwargs # Pass optional arguments
        )
    
    def _create_individual_dataloaders(self) -> List[DataLoader]:
        return [
            DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    worker_init_fn=zarrdataset_worker_init_fn,
                    pin_memory=True,
                )
            for dataset in self.individual_datasets
        ]

    def get_dataloader(self) -> DataLoader:
        """
        Get the DataLoader instance for the combined datasets.

        Returns
        -------
        DataLoader
            DataLoader instance for batch processing.
        """
        return self.dataloader

    def get_individual_datasets(self) -> List[ZarrDataset]:
        """
        Get the list of individual ZarrDataset instances.

        Returns
        -------
        List[ZarrDataset]
            List of individual dataset instances.
        """
        return self.individual_datasets

    def __len__(self) -> int:
        """
        Get the total number of samples across all datasets.

        Returns
        -------
        int
            Total number of samples.
        """
        return len(self.individual_datasets)

    def __getitem__(self, idx):
        """
        Override to get item from the combined dataset.
        """
        sample = super().__getitem__(idx)
        
        if self.transform:
            sample = self.transform({"image": sample})["image"]
        
        sample = np.squeeze(sample)
        return sample.astype(np.float32)
