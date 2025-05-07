"""
Example to instantiate a chain of iterable datasets
"""
import os
import yaml
import numpy as np
from pathlib import Path

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
    #if dist.is_available() and dist.is_initialized():
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
    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")
    print(f"Using device: {rank}/{world_size-1} (Local rank: {local_rank})")

    # load example configuration file
    config_path = Path(__file__).resolve().parent.parent / "configs" / "example_data.yaml"
    cfg = load_config(config_path)
    print(f"Configuration: {cfg}")

    # parse dataset paths
    dataset_paths = [f"{cfg['dataset']['bucket_path']}/{i}" for i in cfg["dataset"]["train"]["paths"]]
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
        )

        # Get the DataLoader
        dataloader = zarr_datasets.get_dataloader()

        # Example of processing a few batches
        print("\nProcessing first 2 batches:")
        for i, batch in enumerate(dataloader):
            if i >= 2:  # Process only 2 batches for demonstration
                break

            # Unpack batch data based on ZarrDataset's return structure
            worker_ids, positions, data = batch

            print(f"\n\tBatch {i+1}:")
            print(f"\tRank: {rank}/{world_size}")
            print(f"\tImages shape: {data.shape}")
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
