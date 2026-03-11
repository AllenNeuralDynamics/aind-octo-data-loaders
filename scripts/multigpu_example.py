import logging
import multiprocessing as mp
import os

import numpy as np
import torch
import torch.distributed as dist
import torchvision
import zarrdataset as zds
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from aind_octo_data_loaders.dataloader import ZarrDatasets


def setup():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def setup_logger():
    logger = logging.getLogger("my_logger")  # You can name it anything
    logger.setLevel(logging.INFO)  # Set the log level

    # Create console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter and attach to handler
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    ch.setFormatter(formatter)

    # Add handler to logger
    logger.addHandler(ch)

    # Example usage
    logger.info("Logger initialized")
    return logger


def main():
    bucket_path = "s3://aind-open-data"
    dataset_paths = [
        "HCR_704576_2024-04-22_13-00-00/SPIM.ome.zarr/R0_X_0000_Y_0003_Z_0000_ch_405.zarr",  # HCR
        "SmartSPIM_722649_2025-04-08_13-01-09_stitched_2025-04-09_06-15-07/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr",  # SmartSPIM
        "HCR_785830_2025-03-19_17-00-00/SPIM/Tile_X_0001_Y_0029_Z_0000_ch_488.ome.zarr",  # Proteomics
    ]

    for i, path in enumerate(dataset_paths):
        dataset_paths[i] = f"{bucket_path}/{path}"
        print(f"Dataset {i+1} path: {dataset_paths[i]}")

    print("Using datasets: ", dataset_paths)

    # Example configuration
    dataset_scales = ["3", "3", "3"]  # Different scales for each dataset
    patch_size = [64, 64, 64]  # Z, Y, X dimensions
    batch_size = 4

    transforms = torchvision.transforms.Compose(
        [zds.ToDtype(dtype=np.float16)]
    )

    # Getting local ranks
    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    # Setting up logger
    logger = setup_logger()

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

    for i, (worker_ids, positions, batch) in enumerate(dataloader):
        if i >= 2:  # Process only 2 batches for demonstration
            break

        batch = batch.to(device)
        batch_memory_bytes = batch.element_size() * batch.nelement()
        batch_memory_gb = batch_memory_bytes / (1024**3)
        print(
            f"[Rank {rank}] Batch {i} | Volume shape: {batch.shape} | Memory: {batch_memory_gb:.2f}GB"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
