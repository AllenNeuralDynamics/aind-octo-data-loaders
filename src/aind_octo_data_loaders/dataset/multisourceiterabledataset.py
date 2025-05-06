from torch.utils.data import IterableDataset


class MultiSourceIterableDataset(IterableDataset):
    def __init__(self, datasets):
        self.datasets = datasets

    def __iter__(self):
        for ds in self.datasets:
            yield from iter(ds)
