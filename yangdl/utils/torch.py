import os
import random
from typing import Optional

import numpy as np
import torch


__all__ = [
    'set_seed',
    'collate_fn'
]


def set_seed(seed: Optional[int] = None):
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


def collate_fn(batch):
    return list(zip(*batch))

