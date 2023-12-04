import os
import random
import shutil
from typing import Optional

import numpy as np
import torch

from yangdl.utils.python import get_caller_file_name


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
            self._set_seed(val)

    def __getattr__(self, key):
        return None

    @staticmethod
    def _set_seed(seed: Optional[int] = None):
        if seed is None:
            return
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


env = Env()

