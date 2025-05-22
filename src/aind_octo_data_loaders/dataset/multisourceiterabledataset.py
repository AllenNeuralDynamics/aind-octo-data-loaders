"""
Declaration of the MultiSourceIterableDataset
"""
from torch.utils.data import IterableDataset
from typing import List


class MultiSourceIterableDataset(IterableDataset):
    """
    Class that combines multiple datasets into one iterable dataset.
    """
    def __init__(self, datasets: List[IterableDataset]):
        """
        Initialize the MultiSourceIterableDataset.
        Parameters
        ----------
        datasets : list of IterableDataset
            List of datasets to combine.
        """
        self.datasets = datasets

    def __iter__(self):
        """
        Iterator over the combined datasets.
        """
        for ds in self.datasets:
            yield from iter(ds)
