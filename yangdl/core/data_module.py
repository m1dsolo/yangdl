from typing import Generator

from torch.utils.data import DataLoader


__all__ = [
    'DataModule',
]


class DataModule():
    """Generate `DataLoader` for each fold.

    You should inherit `DataModule` and override at least one method in
    `{train, val, test, predict}_loader` according to your own task.
    """
    def __init__(self):
        pass

    def train_loader(self) -> Generator[DataLoader, None, None]:
        while True:
            yield None

    def val_loader(self) -> Generator[DataLoader, None, None]:
        while True:
            yield None

    def test_loader(self) -> Generator[DataLoader, None, None]:
        while True:
            yield None

    def predict_loader(self) -> DataLoader:
        while True:
            yield None

