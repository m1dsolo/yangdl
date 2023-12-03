import logging

from rich.logging import RichHandler
import torch

from yangdl.core.env import env
from yangdl.utils.helper import clear_markup


__all__ = [
    'logger',
]


class CombinationHandler(logging.Handler):
    """`RichHandler` for logging to the terminal and `FileHandler` for outputting to the file."""
    def __init__(self, file_name: str = '/dev/null'):
        super().__init__()

        self.rich_handler = RichHandler(
            level=logging.NOTSET,
            show_path=False,
            markup=True,
            rich_tracebacks=True,
            log_time_format='%H:%M:%S'
        )

        self.file_handler = logging.FileHandler(filename=file_name, mode='w', delay=True)
        self.file_handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.file_handler.setLevel(logging.NOTSET)

    def emit(self, record):
        self.rich_handler.emit(record)
        if self.file_handler.baseFilename != '/dev/null':
            record.msg = clear_markup(record.msg)
            self.file_handler.emit(record)

    def set_file_name(self, file_name: str) -> None:
        self.file_handler.baseFilename = file_name


class Logger:
    """Encapsulates logging module (Use singleton `logger` instead of instantiating `Logger`).

    Mainly used to print and record logs.
    """
    def __init__(self, file_name: str = '/dev/null'):
        """
        Args:
            file_name: File to save logs.
        """
        self.logger = logging.getLogger('root')
        self.logger.setLevel(logging.NOTSET)

        self.handler = CombinationHandler(file_name)
        self.logger.addHandler(self.handler)

    def info(self, msg: str, *args, **kwargs):
        self.logger.info(msg, *args, **kwargs, extra={'markup': True})

    def log_props(self, fmt='{:.3f}', **kwargs):
        """Log properties.

        Examples:
            >>> logger.log_props(loss=self.loss.val, auc=self.metric.auc)
            {stage}: fold: {fold}, epoch: {epoch}: loss: 0.3, auc: 0.7
        """
        stage, fold, epoch = env.stage, env.fold, env.epoch
        color = {'train': 'red', 'val': 'blue', 'test': 'green', 'predict': 'pink'}[stage]

        s = f'[{color}]{stage:>5}[/{color}]: fold:{fold:>2}, epoch: {epoch:>2}: '
        for key, val in kwargs.items():
            if isinstance(val, torch.Tensor):
                val = val.cpu().item()
            s += f'{key}: [{color}]{fmt.format(val)}[/{color}], '
        s = s[:-2]

        self.logger.info(s, extra={'markup': True, 'highlighter': None})


logger = Logger()

