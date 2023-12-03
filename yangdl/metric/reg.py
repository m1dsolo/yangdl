import torch
from torch import Tensor

from yangdl.metric import Metric


class RegMetric(Metric):
    """The metric of regression task."""
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.add_tensor('preds', torch.tensor([], dtype=torch.float32))
        self.add_tensor('labels', torch.tensor([], dtype=torch.float32))

        self.properties = ['mae', 'rmse', 'r2']

    @torch.no_grad()
    def update(self, preds: Tensor, labels: Tensor):
        """Update metric with probabilities and labels.

        Args:
            preds: value predicted by the model. Shape is (B,).
                
            labels: ground truth. Shape is (B,)
        """
        assert len(preds) == len(labels) and len(preds.shape) == 1, f'preds.shape={preds.shape}, labels.shape={labels.shape}'

        preds, labels = preds.detach(), labels.detach()

        self.preds = torch.cat([self.preds, preds], dim=0)
        self.labels = torch.cat([self.labels, labels], dim=0)

    @property
    def mae(self) -> Tensor:
        return (self.labels - self.preds).abs().mean()

    @property
    def rmse(self) -> Tensor:
        return ((self.labels - self.preds) ** 2).mean().sqrt()

    @property
    def r2(self) -> Tensor:
        a = ((self.labels - self.labels.mean()) ** 2).sum()
        b = ((self.labels - self.preds) ** 2).sum()

        return 1 - b / a

