"""
Example to instantiate a chain of iterable datasets
"""

import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import yaml

from aind_octo_data_loaders.dataloader import ZarrDatasets


def custom_transform(sample: np.ndarray) -> np.ndarray:
    """
    Example transform function that normalizes the image data.

    Parameters
    ----------
    sample : np.ndarray
        Input sample data.

    Returns
    -------
    np.ndarray
        Normalized sample data.
    """

    if sample.max() > sample.min():
        return (sample - sample.min()) / (sample.max() - sample.min())
    return sample


def load_config(path):
    """load yaml configuration"""
    with open(path, "r") as f:
        return yaml.safe_load(f)


def setup():
    """
    DDP setup to establish local and global ranks following `torchrun` call

    Returns
    -------
    int
        Process rank
    int
        Process world size
    int
        Process local rank
    """
    # if dist.is_available() and dist.is_initialized():
    if True:
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank, local_rank, world_size = 0, 0, 1
    return rank, world_size, local_rank


def main():
    """
    Example usage of the ZarrDatasets class.
    """
    # setup multi-device ranking
    # rank, world_size, local_rank = setup()
    rank, local_rank, world_size = 0, 0, 1
    device = torch.device(f"cuda:{local_rank}")
    print(f"Using device: {rank}/{world_size-1} (Local rank: {local_rank})")

    # load example configuration file
    config_path = (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "example_data.yaml"
    )
    cfg = load_config(config_path)
    print(f"Configuration: {cfg}")

    # parse dataset paths
    dataset_paths = [
        (
            "s3://aind-open-data/SmartSPIM_774928_2024-12-17_17-41-54_stitched_2025-01-11_01-02-44/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr",
            # "s3://aind-msma-morphology-data/test_data/SmartSPIM/smartspim_segmentation_masks/SmartSPIM_774928/segmentation_mask.zarr",
            None,
        ),
        (
            "s3://aind-open-data/SmartSPIM_764220_2025-01-30_11-15-58_stitched_2025-03-06_10-04-25/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr",
            # "s3://aind-msma-morphology-data/test_data/SmartSPIM/smartspim_segmentation_masks/SmartSPIM_764220/segmentation_mask.zarr"
            None,
        ),
        (
            "s3://aind-open-data/SmartSPIM_782499_2025-03-06_00-01-19_stitched_2025-03-07_05-11-31/image_tile_fusing/OMEZarr/Ex_639_Em_680.zarr",
            # "s3://aind-msma-morphology-data/test_data/SmartSPIM/smartspim_segmentation_masks/SmartSPIM_782499/segmentation_mask.zarr"
            None,
        ),
    ]

    print("Using datasets: ", dataset_paths)

    # Configure dataset scales
    # TODO: add this to config
    dataset_scales = ["1", "1", "1"]  # Different scales for each dataset

    patch_size = (
        int(cfg["loader"]["patch_size"]),
        int(cfg["loader"]["patch_size"]),
        int(cfg["loader"]["patch_size"]),
    )
    batch_size = cfg["loader"]["batch_size"]

    print(f"Batch size: {batch_size}")
    print(f"Patch size (ZYX): {patch_size}")

    try:
        # Initialize ZarrDatasets with custom transform
        zarr_datasets = ZarrDatasets(
            dataset_paths=dataset_paths,
            dataset_scales=dataset_scales,
            patch_size_zyx=patch_size,
            batch_size=batch_size,
            transform=custom_transform,
            num_workers=2,
            return_positions=True,
            return_worker_id=True,
        )

        # Get the DataLoader
        dataloader = zarr_datasets.get_dataloader()

        # Example of processing a few batches
        print("\nProcessing first 2 batches:")
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Process only 2 batches for demonstration
                break

            data_batch = batch["data"]
            wavelength_nm = batch["wavelength_nm"]
            numerical_aperture = batch["numerical_aperture"]
            image_resolution = batch["image_resolution"]

            # Unpack batch data based on ZarrDataset's return structure
            worker_ids, positions, data = data_batch

            print(f"\n\tBatch {i+1}:")
            print(f"\tRank: {rank}/{world_size}")
            print(f"\tImages shape: {data.shape}")
            print(f"\tImage wavelength: {wavelength_nm}")
            print(f"\tImage NA: {numerical_aperture}")
            print(f"\tImage resolution: {image_resolution}")
            print(f"\tPositions: {positions}")
            print(f"\tWorker IDs: {worker_ids}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "This example requires actual Zarr datasets. Replace the paths with valid ones."
        )
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
