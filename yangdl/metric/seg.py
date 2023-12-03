from typing import Optional, Sequence

import torch
from torch import Tensor

from yangdl.metric.confusion_matrix import ConfusionMatrix


__all__ = [
    'SegMetric',
]


class SegMetric(ConfusionMatrix):
    """The metric of segmentation task (See base class `ConfusionMatrix` for more information)."""
    def __init__(
        self, 
        num_classes: int,
        properties: Optional[Sequence[str]] = None,
        thresh: Optional[float | str] = None, 
        ignore_label: Optional[int] = None,
        eps: float = 1e-7, 
        **kwargs,
    ):
        """
        Args:
            num_classes: The number of classes.
            properties: The properties you are interested in will be
                automatically logged in tensorboard and printed and saved as results.
                If is `None`, will save all properties
                (check the code to see what properties are there).
            thresh: The threshold that determines whether a sample is positive or negative.
                Only useful in binary classification.
                If is `float`, the value must in [0, 1].
                If is 'f1_score', will use threshold that has best f1_score.
                If is 'roc', will use threshold that has best tpr - fpr.
            ignore_label: Value in [0, C], this label will not used to calculate metrics.
            eps: Prevent division by 0.
        """
        if properties is None:
            if num_classes == 2:
                properties = ['acc', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score', 'dice', 'iou', 'thresh']
            else:
                properties = ['acc', 'precision', 'recall', 'sensitivity', 'specificity', 'f1_score', 'dice', 'iou']
        super().__init__(num_classes, properties, thresh, ignore_label, eps, **kwargs)

    @torch.no_grad()
    def update(
        self,
        probs: Tensor,
        labels: Tensor,
    ):
        """Update metric with probabilities and labels.

        Args:
            probs: probability predicted by the model.
                Shape is (B, [D], H, W) or (B, C, [D], H, W). Value is in [0, 1].
            labels: ground truth.
                Shape is (B, [D], H, W). Value is in (0, 1, ..., C - 1).
        """
        probs = probs.detach()
        if len(probs.shape) > len(labels.shape):
            probs = probs.moveaxis(1, -1)
            probs = probs.flatten(0, -2)
        else:
            probs = probs.flatten(0, -1)
        labels = labels.flatten(0, -1)

        super().update(probs, labels)

