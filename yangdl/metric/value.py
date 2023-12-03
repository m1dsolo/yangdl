import torch
from torch import Tensor

from yangdl.metric import Metric


class ValueMetric(Metric):
    """Accumulate a single value."""
    def __init__(
        self,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.add_tensor('_val', torch.tensor(0, dtype=torch.float))
        self.add_tensor('n', torch.tensor(0, dtype=torch.long))

        self.properties = ['val']

    @torch.no_grad()
    def update(self, val: Tensor, n: int = 1):
        """Update metric with `n` numbers whose average is `val`.

        A common use case is to accumulate loss.
        In this case, `val` is average loss of the entire batch,
        and n is the number of samples in the batch.

        Args:
            val: Average of data.
            n: Length of data.
        """
        val = val.detach()
        if isinstance(val, Tensor):
            val = val.item()

        self._val += val * n
        self.n += n

    @property
    def val(self) -> Tensor:
        if self.n == 0:
            return torch.tensor(-1.)
        return self._val / self.n
