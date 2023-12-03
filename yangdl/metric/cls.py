from typing import Optional

from yangdl.metric.confusion_matrix import ConfusionMatrix


__all__ = [
    'ClsMetric',
]


class ClsMetric(ConfusionMatrix):
    """The metric of classification task (See base class `ConfusionMatrix` for more information)."""
    def __init__(
        self, 
        num_classes: int,
        properties: Optional[list[str]] = None,
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
                properties = ['acc', 'auc', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity', 'ap', 'thresh']
            else:
                properties = ['acc', 'auc', 'precision', 'recall', 'f1_score', 'sensitivity', 'specificity']
        super().__init__(num_classes, properties, thresh, ignore_label, eps, **kwargs)


