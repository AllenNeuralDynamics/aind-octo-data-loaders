import torch
import numpy as np
import zarr
import tqdm
import boto3
import matplotlib.pyplot as plt
import zarr
import numpy as np

class Mock3DDataset(torch.utils.data.Dataset):
    def __init__(self, size=100, shape=(64, 64, 64)):
        self.size = size
        self.shape = shape

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        volume = np.random.rand(*self.shape).astype(np.float32)
        label = np.random.randint(0, 2)
        return torch.tensor(volume), torch.tensor(label), idx


def open_img(prefix):
    """
    Opens an image stored in an S3 bucket as an N5 array.

    Parameters:
    -----------
    prefix : str
        Prefix (or path) within the S3 bucket where the image is stored.

    Returns:
    --------
    zarr.core.Array
        A Zarr object representing the image data.

    """
    return zarr.open(zarr.N5FSStore(prefix))


def load_dataset(bucket_name, prefix):
    """
    Loads a dataset from an S3 bucket, given the bucket name and prefix, and
    returns a dictionary containing input images and corresponding label masks
    for each example.

    Parameters
    ----------
    bucket_name : str
        Name of the S3 bucket where the dataset is stored.
    prefix : str 
        Prefix to list the dataset.

    Returns
    -------
    dict
        Dictionary where keys are tuples (brain_id, block_id), and values are
        dictionaries with the following structure:
            - "input": input image.
            - "label_mask": segmentation mask.

    """
    dataset = dict()
    for brain_prefix in tqdm(list_s3_prefixes(bucket_name, prefix)):
        brain_id = brain_prefix.split("/")[-2].split(".")[0]
        brain_prefix += "images/" if "test" in brain_prefix else ""
        for block_prefix in list_s3_prefixes(bucket_name, brain_prefix):
            # Get image prefixes
            block_id = block_prefix.split("/")[-2]
            input_prefix = f"s3://{bucket_name}/{block_prefix}input.n5/"
            label_prefix = f"s3://{bucket_name}/{block_prefix}label_mask.n5/"
            print(f"Input Prefix: {input_prefix}")
            # Populate dataset
            key = (brain_id, block_id)
            dataset[key] = {
                "input": open_img(input_prefix),
                "label_mask": open_img(label_prefix)
            }
    return dataset


def list_s3_prefixes(bucket_name, prefix):
    """
    Lists all immediate subdirectories of a given S3 prefix (path).

    Parameters
    -----------
    bucket_name : str
        Name of the S3 bucket to search.
    prefix : str
        S3 prefix (path) to search within.

    Returns:
    --------
    List[str]
        List of immediate subdirectories under the given prefix.

    """
    # Check prefix is valid
    if not prefix.endswith("/"):
        prefix += "/"

    # Call the list_objects_v2 API
    s3 = boto3.client("s3")
    response = s3.list_objects_v2(
        Bucket=bucket_name, Prefix=prefix, Delimiter="/"
    )
    if "CommonPrefixes" in response:
        return [cp["Prefix"] for cp in response["CommonPrefixes"]]
    else:
        return list()


def plot_mips(img):
    """
    Plots the Maximum Intensity Projections (MIPs) of a 3D image along the XY,
    XZ, and YZ axes.

    Parameters
    ----------
    img : ArrayLike
        Input 3D image to generate MIPs from.

    Returns
    -------
    None

    """
    fig, axs = plt.subplots(1, 3, figsize=(10, 4))
    axs_names = ["XY", "XZ", "YZ"]
    p = np.percentile(img, 99.9)
    for i in range(3):
        axs[i].imshow(np.clip(np.max(img, axis=i), 0, p), cmap="grey")
        axs[i].set_title(axs_names[i], fontsize=16)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
    plt.tight_layout()
    plt.show()
