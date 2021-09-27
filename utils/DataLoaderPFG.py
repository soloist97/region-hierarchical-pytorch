from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator


__all__ = [
    "DataLoaderPFG"
]


class DataLoaderPFG(DataLoader):
    """
    Prefetch version of DataLoader: https://github.com/IgorSusmelj/pytorch-styleguide/issues/5
    """

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
