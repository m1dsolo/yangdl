import os
import shutil

from yangdl.utils.python import get_caller_file_name
from yangdl.utils.torch import set_seed


__all__ = [
    'env',
]


class Env:
    """Environmental context of `yangdl` (Use singleton `env` instead of instantiating `Env`).

    Used to initialize `yangdl` configuration and manage global variables.
    """
    # set these manually
    exp_path: str = None  # experiment directory. Set to `None` generally for `predict` task.
    seed: int = None  # it is best to set seed for reproduction

    # Automatically generated values:
    stage: str = None # Value in {'train', 'val', 'test', 'predict'}.
    fold: int = None # The count of current fold.
    epoch: int = None # The count of current epoch.
    step: int = None # The count of current step.

    def __setattr__(self, key, val):
        self.__dict__[key] = val

        if key == 'exp_path':
            for file_name in ('log', 'metric', 'ckpt'):
                os.makedirs(f'{val}/{file_name}', exist_ok=True)
            shutil.copyfile(get_caller_file_name(), f'{val}/code.py')

            from yangdl.core.logger import logger
            logger.handler.set_file_name(f'{val}/log/log.txt')
        elif key == 'seed':
            set_seed(val)

    def __getattr__(self, key):
        return None


env = Env()

