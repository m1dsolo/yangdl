import torch
from torch import nn
from torch.nn.modules import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from yangdl.metric import Metric


__all__ = [
    'ModelModule',
]


class ModelModule():
    """Encapsulate models and task implementation.

    You should inherit `ModelModule` and override at least one method in `{train, val, test, predict}_step` according to your own task.
    """
    def __init__(self):
        self._registered_models = {}

    def __iter__(self):
        """Called on each new fold.

        You don't need to yield anything, you mainly need to initialize a new model for each fold.
        """
        yield

    def train_step(self, batch) -> None:
        """Overridable train function called at each step."""
        pass

    def val_step(self, batch) -> None:
        """Overridable val function called at each step."""
        pass

    def test_step(self, batch) -> None:
        """Overridable test function called at each step."""
        pass

    def predict_step(self, batch) -> dict:
        """Overridable predict function called at each step."""
        pass

    def train_epoch_begin(self) -> None:
        """Overridable callback function at the begin of epoch in the train stage."""
        pass

    def val_epoch_begin(self) -> None:
        """Overridable callback function at the begin of epoch in the val stage."""
        pass

    def test_epoch_begin(self) -> None:
        """Overridable callback function at the begin of epoch in the test stage."""
        pass

    def predict_epoch_begin(self) -> None:
        """Overridable callback function at the begin of epoch in the predict stage."""
        pass

    def train_epoch_end(self) -> None:
        """Overridable callback function at the end of epoch in the train stage."""
        pass

    def val_epoch_end(self) -> None:
        """Overridable callback function at the end of epoch in the val stage."""
        pass

    def test_epoch_end(self) -> None:
        """Overridable callback function at the end of epoch in the test stage."""
        pass

    def predict_epoch_end(self) -> None:
        """Overridable callback function at the end of epoch in the predict stage."""
        pass
    
    def register_models(self, d) -> None:
        """
        Models that are not `nn.Module` subclasses must be registered.
        The registered models will be automatically moved to gpu.
        The model must have `cuda` method.
        If the `ModelModule.save_ckpt` and `ModelModule.load_ckpt` methods are not overriden,
        the model must have `state_dict` and `load_state_dict` methods.
        """
        self._registered_models.update(d)

    @property
    def named_models(self) -> dict[str, nn.Module]:
        return dict(filter(lambda kv: Module in kv[1].__class__.__mro__, self.__dict__.items())) | self._registered_models

    @property
    def named_metrics(self) -> dict[str, Metric]:
        return dict(filter(lambda kv: Metric in kv[1].__class__.__mro__, self.__dict__.items()))

    @property
    def named_optimizers(self) -> dict[str, Optimizer]:
        return dict(filter(lambda kv: Optimizer in kv[1].__class__.__mro__, self.__dict__.items()))

    @property
    def named_schedulers(self) -> dict[str, _LRScheduler]:
        return dict(filter(lambda kv: _LRScheduler in kv[1].__class__.__mro__, self.__dict__.items()))

    def cuda(self):
        """Move all models and metrics in `ModelModule` to gpu.This method modifies `ModelModule` in-place.
        """
        for model in self.named_models.values():
            model.cuda()
        for metric in self.named_metrics.values():
            metric.cuda()

        return self

    def save_ckpt(self, ckpt_name: str):
        """Save all models and optimizers' state_dict to ckpt_name.

        TODO: save lr_schedulers.
        """
        ckpt = {}
        for name, model in self.named_models.items():
            ckpt[name] = model.state_dict()
        for name, optimizer in self.named_optimizers.items():
            ckpt[name] = optimizer.state_dict()

        torch.save(ckpt, ckpt_name)

    def load_ckpt(self, ckpt_name: str):
        """Load all models and optimizers' state_dict from ckpt_name."""
        ckpt = torch.load(ckpt_name)
        for name, state_dict in ckpt.items():
            getattr(self, name).load_state_dict(state_dict)

    @property
    def device(self) -> torch.device:
        """
        Currently the framework only supports a single gpu.
        So this method will only return a single device.
        """
        for model in self.named_models.values():
            for param in model.parameters():
                return param.device

