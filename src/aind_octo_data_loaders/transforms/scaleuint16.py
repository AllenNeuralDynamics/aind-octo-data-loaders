import numpy as np
import torch

class ScaleUint16ToFloat3D:
    def __init__(self, method="global", eps=1e-8):
        """
        Parameters
        ----------
        method : str
            - "global": scales from [0, 65535] to [0.0, 1.0]
            - "minmax": scales each volume to [0.0, 1.0] based on its own min/max
            - "zscore": standardizes each volume to zero mean, unit variance
        eps : float
            Small constant to avoid division by zero
        """
        self.method = method
        self.eps = eps

    def __call__(self, volume: np.ndarray) -> torch.Tensor:
        #assert volume.dtype == np.uint16, "Expected uint16 input"

        vol = volume.astype(np.float32)

        if self.method == "global":
            vol = vol / 65535.0

        elif self.method == "minmax":
            vol = (vol - vol.min()) / (vol.max() - vol.min() + self.eps)

        elif self.method == "zscore":
            vol = (vol - vol.mean()) / (vol.std() + self.eps)

        else:
            raise ValueError(f"Unsupported normalization method: {self.method}")

        return torch.from_numpy(vol)
