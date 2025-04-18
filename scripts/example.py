"""
Example to instantiate a chain of iterable datasets
"""
import os
import numpy as np

import torch
import torch.distributed as dist

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

def setup():
    if dist.is_available() and dist.is_initialized():
        dist.init_process_group("nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        local_rank = 0
        world_size = 1
    return rank, world_size, local_rank


def main():
    """
    Example usage of the ZarrDatasets class.
    """

    bucket_path = "s3://aind-open-data"
    dataset_paths = [
        "HCR_704576_2024-04-22_13-00-00/SPIM.ome.zarr/R0_X_0000_Y_0003_Z_0000_ch_405.zarr",  # HCR
        "SmartSPIM_722649_2025-04-08_13-01-09_stitched_2025-04-09_06-15-07/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr",  # SmartSPIM
        "HCR_785830_2025-03-19_17-00-00/SPIM/Tile_X_0001_Y_0029_Z_0000_ch_488.ome.zarr",  # Proteomics
    ]

    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    for i, path in enumerate(dataset_paths):
        dataset_paths[i] = f"{bucket_path}/{path}"
        print(f"Dataset {i+1} path: {dataset_paths[i]}")

    print("Using datasets: ", dataset_paths)
    # Example configuration
    dataset_scales = ["3", "3", "3"]  # Different scales for each dataset
    patch_size = [64, 64, 64]  # Z, Y, X dimensions
    batch_size = 4

    try:
        # Initialize ZarrDatasets with custom transform
        zarr_datasets = ZarrDatasets(
            dataset_paths=dataset_paths,
            dataset_scales=dataset_scales,
            patch_size_zyx=patch_size,
            batch_size=batch_size,
            transform=custom_transform,
            num_workers=2,
        )

        # Get the DataLoader
        dataloader = zarr_datasets.get_dataloader()

        print(f"Batch size: {batch_size}")
        print(f"Patch size (ZYX): {patch_size}")

        # Example of processing a few batches
        print("\nProcessing first 2 batches:")
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Process only 2 batches for demonstration
                break

            # Unpack batch data based on ZarrDataset's return structure
            worker_ids, positions, data = batch

            print(f"Batch {i+1}:")
            print(f"Rank: {rank}/{world_size}")
            print(f"  Images shape: {data.shape}")
            print(f"  Positions: {positions}")
            print(f"  Worker IDs: {worker_ids}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(
            "This example requires actual Zarr datasets. Replace the paths with valid ones."
        )
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
