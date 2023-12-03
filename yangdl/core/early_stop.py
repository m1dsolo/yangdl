import numpy as np


__all__ = [
    'EarlyStop',
]


class EarlyStop():
    """Early stop in `train` stage based on performance during the `val` stage."""
    def __init__(
        self,
        monitor: dict[str] = {'loss.val': 'small'},
        delta: float = 0.,
        patience: int = 10,
        min_stop_epoch: int = 10,
        max_stop_epoch: int = 100,
    ):
        """
        Args:
            monitor: Monitored metric used to decide whether to early stop.
                Examples: {'loss.loss': 'small'}: Model will early stop with the lowest loss.
                {'metric.auc': 'big'}: Model will early stop with the highest auc.
            delta: The performance will be considered better only if the result is at least `delta` better than the past.
            patience: If the performance is not better in the `val` stage for consecutive `patience` epochs,
                `train` will be stopped.
            min_stop_epoch: Will not early stop when epoch is less than `min_stop_epoch`.
            max_stop_epoch: Train will be forced to stop at `max_stop_epoch`.
        """
        self.metric_name, self.rule = next(iter(monitor.items()))
        self.delta = delta
        self.patience = patience
        self.min_stop_epoch = min_stop_epoch
        self.max_stop_epoch = max_stop_epoch
        self.res = {'stop_epoch': [], 'best_epoch': []}

        self.init()

    def init(self):
        self.counter = 0
        self.best_val = {'small': np.Inf, 'big': -np.Inf}[self.rule]
        self.stop_epoch = 0
        self.best_epoch = 0
        self.early_stop = False

    def __call__(self, val, epoch):
        if self.rule == 'small':
            if val < self.best_val - self.delta:
                self.best_val = val
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1
        elif self.rule == 'big':
            if val > self.best_val + self.delta:
                self.best_val = val
                self.counter = 0
                self.best_epoch = epoch
            else:
                self.counter += 1

        if self.counter > self.patience and epoch >= self.min_stop_epoch or epoch == self.max_stop_epoch:
            self.stop_epoch = epoch
            self.early_stop = True
            self.res['stop_epoch'].append(self.stop_epoch)
            self.res['best_epoch'].append(self.best_epoch)

    def to_dict(self):
        return self.res

