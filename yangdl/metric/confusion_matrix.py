import torch
from torch import Tensor
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from typing import Optional

from yangdl.metric import Metric


__all__ = [
    'ConfusionMatrix',
]


class ConfusionMatrix(Metric):
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
                If `num_classes == 2`, the negative class(label is equal to 0) does not participate in the metric calculation.
            properties: The properties you are interested in will be
                automatically logged in tensorboard and printed and saved as results.
                If is None, will save all properties
                (check the code to see what properties are there).
            thresh: The threshold that determines whether a sample is positive or negative.
                Only useful in binary classification.
                If is float, the value range is between 0 and 1.
                If is 'f1_score', will use threshold that has best f1_score.
                If is 'roc', will use threshold that has best tpr - fpr.
            ignore_label: Value in [0, C], this label will not use to calculate metrics.
            eps: Prevent division by 0.
        """
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.properties = properties
        if thresh is None and num_classes == 2:
            thresh = 0.5
        self._thresh = thresh
        self.ignore_label = ignore_label

        self.add_tensor('_matrix', torch.zeros((num_classes, num_classes), dtype=torch.long))
        self.eps = eps
        self.save_probs = 'auc' in properties or 'ap' in properties or isinstance(thresh, str)

        if self.save_probs:
            self.add_tensor('probs', torch.tensor([], dtype=torch.float)) # num_classes == 2: (B,), else: (B, C)
            self.add_tensor('labels', torch.tensor([], dtype=torch.long)) # (B,)

    @torch.no_grad()
    def update(
        self,
        probs: Tensor,
        labels: Tensor,
    ) -> None:
        """Update metric with probabilities and labels.

        Args:
            probs: probability predicted by the model.
                Shape is (B,) or (B, C). Value is in [0, 1].
            labels: ground truth.
                Shape is (B,). Value is in (0, 1, ..., C - 1).
        """
        probs = probs.detach()
        if self.num_classes == 2 and len(probs.shape) > len(labels.shape):
            probs = probs[:, -1]
        if self.ignore_label is not None:
            mask = labels != self.ignore_label
            probs = probs[mask]
            labels = labels[mask]

        if not isinstance(self._thresh, str):
            self._matrix += self._calc_matrix(probs, labels)

        if self.save_probs:
            self.probs = torch.cat([self.probs, probs], dim=0)
            self.labels = torch.cat([self.labels, labels], dim=0)

    def _calc_matrix(
        self,
        probs: Tensor,
        labels: Tensor,
    ) -> Tensor:
        preds = self._calc_preds(probs)
        bins = self._bincount(labels * self.num_classes + preds, minl=self.num_classes ** 2)
        return bins.reshape(self.num_classes, self.num_classes)
    
    def _calc_preds(
        self,
        probs: Tensor,
    ) -> Tensor:
        if self.num_classes == 2:
            return (probs > self.thresh).long()
        else:
            return probs.argmax(dim=1).long()

    def _bincount(self, x: Tensor, minl: Optional[int] = None):
        """
        Example: x=[1, 1, 0, 2, 2] --> res=[1, 2, 2].
        """
        if minl is None:
            minl = x.max() + 1
        res = torch.empty(minl, device=x.device, dtype=torch.long)
        for i in range(minl):
            res[i] = (x == i).sum()
        return res

    def ravel(self, idx: Optional[int] = None):
        """Convert confusion matrix to (tn, fp, fn, tp).

        Args:
            idx: Positive class idx.
        """
        tp = self.matrix[idx, idx]
        fp = self.matrix[:, idx].sum() - tp
        fn = self.matrix[idx, :].sum() - tp
        tn = self.matrix.sum() - tp - fp - fn

        return tn, fp, fn, tp

    @property
    def matrix(self):
        if isinstance(self._thresh, str) and self._matrix.sum() != len(self.probs):
            self._matrix = self._calc_matrix(self.probs, self.labels)
        return self._matrix

    @property
    def acc(self):
        return self.matrix.diagonal().sum() / (self.matrix.sum() + self.eps)

    @property
    def precision(self):
        return self.precisions.mean()
    
    @property
    def precisions(self):
        precisions = []
        for idx in range(self.num_classes == 2, self.num_classes):
            tn, fp, fn, tp = self.ravel(idx)
            precisions.append(tp / (tp + fp + self.eps))
        return torch.stack(precisions, dim=0)
    
    @property
    def recall(self):
        return self.recalls.mean()
    
    @property
    def recalls(self):
        recalls = []
        for idx in range(self.num_classes == 2, self.num_classes):
            tn, fp, fn, tp = self.ravel(idx)
            recalls.append(tp / (tp + fn + self.eps))
        return torch.stack(recalls, dim=0)

    @property
    def sensitivity(self):
        return self.recall

    @property
    def sensitivitys(self):
        return self.recalls

    @property
    def specificity(self):
        return self.specificitys.mean()

    @property
    def specificitys(self):
        specificitys = []
        for idx in range(self.num_classes == 2, self.num_classes):
            tn, fp, fn, tp = self.ravel(idx)
            specificitys.append(tn / (tn + fp + self.eps))
        return torch.stack(specificitys, dim=0)

    @property
    def f1_score(self):
        return 2 * self.precision * self.recall / (self.precision + self.recall + self.eps)

    @property
    def f1_scores(self):
        f1_scores = []
        for precision, recall in zip(self.precisions, self.recalls):
            f1_scores.append(2 * precision * recall / (precision + recall + self.eps))
        return torch.stack(f1_scores, dim=0)

    @property
    def auc(self):
        assert self.save_probs is True
        res = -1.
        try:
            if self.num_classes == 2:
                if self.labels.sum().item() not in (0, len(self.labels)):
                    res = roc_auc_score(self.labels.cpu().numpy(), self.probs.cpu().numpy(), average='macro')
            else:
                res = roc_auc_score(self.labels.cpu().numpy(), self.probs.cpu().numpy(), average='macro', multi_class='ovr')
        except Exception as e:
            print(e)
        return torch.tensor(res, dtype=torch.float)

    @property
    def ap(self):
        assert self.save_probs is True
        assert self.num_classes == 2, f'num_classes={self.num_classes}'

        res = -1.
        if self.labels.sum().item() not in (0, len(self.labels)):
            res = average_precision_score(self.labels.cpu().numpy(), self.probs.cpu().numpy())
        return torch.tensor(res, dtype=torch.float)

    @property
    def dice(self):
        return self.dices.mean()
    
    @property
    def dices(self):
        dices = []
        for idx in range(self.num_classes == 2, self.num_classes):
            tn, fp, fn, tp = self.ravel(idx)
            dices.append(2 * tp / (fp + 2 * tp + fn + self.eps))
        return torch.stack(dices, dim=0)

    @property
    def iou(self):
        return self.ious.mean()

    @property
    def ious(self):
        ious = []
        for idx in range(self.num_classes == 2, self.num_classes):
            tn, fp, fn, tp = self.ravel(idx)
            ious.append(tp / (fp + tp + fn + self.eps))
        return torch.stack(ious, dim=0)
    
    @property
    def thresh(self):
        assert self.num_classes == 2, f'num_classes={self.num_classes}'

        if self._thresh is None:
            thresh = -1.

        if not isinstance(self._thresh, str):
            thresh = self._thresh
        else:
            assert self.save_probs is True
            labels, probs = self.labels.cpu().numpy(), self.probs.cpu().numpy()
            if labels.sum().item() in (0, len(labels)):
                thresh = 0.5
            else:
                if self._thresh == 'f1_score':
                    precisions, recalls, threshs = precision_recall_curve(labels, probs)
                    f1s = 2 * precisions * recalls / (precisions + recalls + self.eps)
                    thresh = threshs[f1s.argmax(axis=0)]
                elif self._thresh == 'roc':
                    fpr, tpr, threshs = roc_curve(labels, probs, pos_label=1)
                    thresh = threshs[(tpr - fpr).argmax(axis=0)]

        return torch.tensor(thresh, dtype=torch.float)

