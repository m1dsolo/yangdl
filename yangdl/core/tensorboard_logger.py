from torch.utils.tensorboard import SummaryWriter

from yangdl.core.env import env
from yangdl.metric import Metric


__all__ = [
    'TensorBoardLogger',
]


class TensorBoardLogger():
    """The logger of tensorboard."""
    def __init__(self, fold: int):
        if env.exp_path is not None:
            self.writer = SummaryWriter(f'{env.exp_path}/log/{fold}')
        else:
            self.writer = None

    def log(
        self, 
        metric: Metric,
        stage: str,
        step: int,
    ):
        """Log `Metric` to tensorboard."""
        if self.writer is None:
            return

        for prop_name, prop in metric.data.items():
            if isinstance(prop, list): # lists are not currently logged
                continue

            if metric.prefix is not None:
                prop_name = f'{metric.prefix}_{prop_name}'
            self.writer.add_scalar(f'{prop_name}/{stage}', prop, step)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()

