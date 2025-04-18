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

mp.set_start_method("spawn", force=True)


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
    DATASET_PATH = "s3://aind-open-data/SmartSPIM_709392_2024-01-29_18-33-39_stitched_2024-02-04_12-45-58/image_tile_fusing/OMEZarr/Ex_639_Em_667.zarr"

    data_scale = "1"
    batch_size = 64
    patch_size = 256

    rank, world_size, local_rank = setup()
    device = torch.device(f"cuda:{local_rank}")

    logger = setup_logger()

    my_patch_sampler = zds.PatchSampler(
        patch_size=dict(X=patch_size, Y=patch_size, Z=patch_size)
    )

    img_preprocessing = torchvision.transforms.Compose(
        [zds.ToDtype(dtype=np.float16)]
    )

    my_dataset = zds.ZarrDataset(
        [
            zds.ImagesDatasetSpecs(
                modality="images",
                filenames=[DATASET_PATH],
                source_axes="TCZYX",
                data_group=data_scale,
                transform=img_preprocessing,
            ),
        ],
        patch_sampler=my_patch_sampler,
        shuffle=True,
        return_positions=True,
        return_worker_id=True,
    )

    my_dataloader = DataLoader(
        my_dataset,
        batch_size=batch_size,
        worker_init_fn=zds.zarrdataset_worker_init_fn,
    )

    for i, (worker_ids, positions, batch) in enumerate(my_dataloader):
        batch = batch.to(device)
        batch_memory_bytes = batch.element_size() * batch.nelement()
        batch_memory_gb = batch_memory_bytes / (1024**3)
        print(
            f"[Rank {rank}] Batch {i} | Volume shape: {batch.shape} | Memory: {batch_memory_gb:.2f}GB"
        )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
