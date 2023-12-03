from copy import deepcopy
from typing import Optional

from torch import Tensor


class Metric():
    """Base metric class.

    Need to be inherited and implement `update` method.
    """
    def __init__(
        self,
        prefix: Optional[str] = None, 
        freq: tuple[str, int] = ('epoch', 1),
    ):
        """
        Args:
            prefix: If is `None`, tensorboard log will use var name 'train/auc'.
                If is `str`, tensorboard log will use var name f'train/{prefix}_auc'.
            freq: The frequency of logging to tensorboard.
                Examples: ('epoch', 1): log every epoch. ('step', 5): log every 5 steps.
        """
        # use for tensorboard logger
        self.prefix = prefix
        self.freq = freq
        self.properties = None

        self.named_tensors = {}

    def add_tensor(
        self, 
        name: str, 
        tensor: Tensor, 
    ) -> None:
        """Add automatically managed tensor.

        Args:
            name: variable name
            tensor: tensor default value
        """
        self.named_tensors[name] = tensor
        setattr(self, name, deepcopy(tensor))

    def cuda(self):
        """Move all tensor and Metric's tensor to cuda (Modified in-place.)."""
        self.named_tensors = {name: tensor.cuda() for name, tensor in self.named_tensors.items()}

        for name, val in vars(self).items():
            if isinstance(val, Tensor):
                setattr(self, name, val.cuda())
            if isinstance(val, Metric):
                val.cuda()

        return self

    def reset(self) -> None:
        """Reset all tensor and `Metric`."""
        for name, tensor in self.named_tensors.items():
            setattr(self, name, deepcopy(tensor))

        for name, val in vars(self).items():
            if isinstance(val, Metric):
                getattr(self, name).reset()

    @property
    def data(self) -> dict:
        """Return all {property_name: property} pair of this `Metric`."""
        return {key: getattr(self, key).cpu().tolist() for key in self.properties}

