"""
Concatenated zarr iterable dataset
"""

from typing import Callable, List, Optional, Union

from torch.utils.data import ChainDataset, DataLoader
from zarrdataset import (
    ImagesDatasetSpecs,
    PatchSampler,
    ZarrDataset,
    chained_zarrdataset_worker_init_fn,
)


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
    ):
        self.dataset_paths = dataset_paths
        self.axes = axes
        self.dataset_scales = dataset_scales
        self.patch_size_zyx = patch_size_zyx
        self.transform = transform
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Ensure dataset_paths and dataset_scales have the same length
        if len(dataset_paths) != len(dataset_scales):
            raise ValueError(
                f"Number of dataset paths ({len(dataset_paths)}) must match "
                f"number of dataset scales ({len(dataset_scales)})"
            )

        self.individual_datasets = []
        self._initialize_datasets()

    def _initialize_datasets(self):
        """Initialize the patch sampler and create the datasets."""
        self.patch_sampler = self._create_patch_sampler()
        self.individual_datasets = self._create_datasets()
        self.zarr_datasets = ChainDataset(self.individual_datasets)
        self.dataloader = self._create_dataloader()

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

    def _create_datasets(self) -> List[ZarrDataset]:
        """
        Create individual ZarrDataset instances for each input path.

        Returns
        -------
        List[ZarrDataset]
            List of initialized ZarrDataset objects.
        """
        zarr_datasets = []
        for i, dataset_path in enumerate(self.dataset_paths):
            # Validate dataset path
            # if not os.path.exists(dataset_path):
            #     raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

            zarr_datasets.append(
                ZarrDataset(
                    dataset_specs=[
                        ImagesDatasetSpecs(
                            filenames=[dataset_path],
                            modality="images",
                            source_axes=self.axes,
                            data_group=str(self.dataset_scales[i]),
                            transform=self.transform,
                        )
                    ],
                    patch_sampler=self.patch_sampler,
                    shuffle=self.shuffle,
                    return_positions=True,
                    return_worker_id=True,
                )
            )

        return zarr_datasets

    def _create_dataloader(self) -> DataLoader:
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
            pin_memory=True,
        )

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
        return len(self.zarr_datasets)
