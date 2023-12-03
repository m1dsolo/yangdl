from collections import defaultdict
import copy
import datetime
import json
import os
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.cuda.amp import autocast
from torch.utils.data import DataLoader

from yangdl.core.model_module import ModelModule
from yangdl.core.data_module import DataModule
from yangdl.core.logger import logger
from yangdl.core.env import env
from yangdl.core.early_stop import EarlyStop
from yangdl.core.tensorboard_logger import TensorBoardLogger
from yangdl.core.ui import EpochProgress
from yangdl.metric import Metric
from yangdl.utils.helper import (
    WithNone,
    apply_format_to_float,
)
from yangdl.utils.io import dict2json
from yangdl.utils.python import method_is_overrided_in_subclass


__all__ = [
    'TaskModule',
]


class TaskModule():
    """Encapsulate the basic process of deep learning.

    You need to pass in parameters to instantiate `TaskModule`,
    and then just run the `do` function.

    Examples:
        >>> task_module = TaskModule(model_module=MyModelModule(), data_module=MyDataModule())
        >>> res = task_module.do()
        {'train': {'metric': {'acc': 0.7, 'auc': 0.8}}}
    """
    def __init__(
        self,
        model_module: ModelModule,
        data_module: DataModule,
        early_stop_params: Optional[dict] = None,
        save_ckpt_period: int = 0,
        val_first: bool = False,
        fmt: str = '{:.4f}',
        benchmark: bool = False,
        deterministic: bool = True,
    ):
        """
        Args:
            model_module: Inherited from `yangdl.ModelModule`.
            data_module: Inherited from `yangdl.DataModule`.
            early_stop_params: Is a params dict to initialize `EarlyStop`. Refer to `EarlyStop` for more details.
            save_ckpt_period: Checkpoints will be saved every `save_ckpt_period` epochs.
                If equal to 0, only `best.pt` will be saved.
            val_first: If set to True, before train at 1 epoch, val will be at 0 epoch.
            fmt: Control the format of output metric values.
            benchmark: Refer to `torch.backends.cudnn.benchmark` for more information.
            deterministic: Refer to `torch.backends.cudnn.deterministic` for more information.
        """

        self.model_module = model_module
        self.data_module = data_module
        self.early_stop = EarlyStop(**early_stop_params) if early_stop_params is not None else None
        self.tensorboard_logger = None
        self.save_ckpt_period = save_ckpt_period
        self.val_first = val_first
        self.fmt = fmt
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic

        self.train: bool = method_is_overrided_in_subclass(data_module.train_loader)
        self.val: bool = method_is_overrided_in_subclass(data_module.val_loader)
        self.test: bool = method_is_overrided_in_subclass(data_module.test_loader)
        self.predict: bool = method_is_overrided_in_subclass(data_module.predict_loader)

        self.train_named_metrics: list[dict[str, Metric]] = []
        self.val_named_metrics: list[dict[str, Metric]] = []
        self.test_named_metrics: list[dict[str, Metric]] = []

    @staticmethod
    def _log_wrapper(func):
        """Log some information for `do` method."""
        def wrapper(self, *args, **kwargs):
            start = datetime.datetime.now()
            res = func(self, *args, **kwargs)
            logger.info(json.dumps(apply_format_to_float(res, self.fmt), indent=2))
            s = (datetime.datetime.now() - start).total_seconds()
            h = s // 3600
            s = s % 3600
            m = s // 60
            s = s % 60
            logger.info('Total time: %d:%02d:%02d', h, m, s)
            logger.info('Experiment path: %s', env.exp_path)
            return res

        return wrapper

    @_log_wrapper
    def do(self) -> None:
        """Begin your multi-fold train, val, test, predict!"""
        for fold, (_, train_loader, val_loader, test_loader, predict_loader) in enumerate(zip(self.model_module, self.data_module.train_loader(), self.data_module.val_loader(), self.data_module.test_loader(), self.data_module.predict_loader()), 1):
            self.model_module.cuda()
            self.tensorboard_logger = TensorBoardLogger(fold)
            if self.early_stop is not None:
                self.early_stop.init()

            self._do_fold(fold, train_loader, val_loader, test_loader, predict_loader)

        metric_path = f'{env.exp_path}/metric'
        res = {}
        if self.train:
            train_res = self._named_metrics2res(self.train_named_metrics)
            dict2json(apply_format_to_float(f'{metric_path}/train.json', self.fmt), train_res)
            res['train'] = train_res
        if self.val:
            val_res = self._named_metrics2res(self.val_named_metrics)
            dict2json(apply_format_to_float(f'{metric_path}/val.json', self.fmt), val_res)
            res['val'] = val_res
        if self.test:
            test_res = self._named_metrics2res(self.test_named_metrics)
            dict2json(apply_format_to_float(f'{metric_path}/test.json', self.fmt), test_res)
            res['test'] = test_res
        if self.early_stop is not None:
            dict2json(f'{metric_path}/early_stop.json', self.early_stop.to_dict())

        return self._simplify_res(res)

    def _do_fold(
        self, 
        fold: Optional[int], 
        train_loader: Optional[DataLoader] = None, 
        val_loader: Optional[DataLoader] = None, 
        test_loader: Optional[DataLoader] = None,
        predict_loader: Optional[DataLoader] = None,
    ) -> None:
        """Process one fold of task.

        One fold can be composed of some of the 4 stages of {train, val, test, predict}.
        The usual order is: train->val->train->val->...->train->val->test->predict.
        """
        self.train_named_metrics.append({})
        self.val_named_metrics.append({})
        self.test_named_metrics.append({})

        if train_loader:
            for epoch in range(0 if self.val_first else 1, self.early_stop.max_stop_epoch + 1):
                if epoch > 0:
                    train_named_metrics = self._do_epoch('train', train_loader, fold, epoch)
                    if self.save_ckpt_period > 0 and epoch > 0 and epoch % self.save_ckpt_period == 0:
                        self._save_ckpt(fold, epoch)

                if val_loader:
                    val_named_metrics = self._do_epoch('val', val_loader, fold, epoch)

                    var_name, val_name = self.early_stop.metric_name.split('.')
                    self.early_stop(getattr(val_named_metrics[var_name], val_name), epoch)
                    if self.early_stop.best_epoch == epoch:
                        self._save_ckpt(fold, None)
                        self.val_named_metrics[-1] = val_named_metrics
                        if epoch > 0:
                            self.train_named_metrics[-1] = train_named_metrics

                    if self.early_stop.early_stop:
                        break
        else:
            if val_loader:
                self.val_named_metrics[-1] = self._do_epoch('val', val_loader, fold, -1)

        if train_loader:
            self._load_ckpt(fold, None)
        if test_loader:
            self.test_named_metrics[-1] = self._do_epoch('test', test_loader, fold, -1)
        if predict_loader:
            self._do_epoch('predict', predict_loader, fold, -1)

    def _do_epoch(
        self,
        stage: str,
        loader: Optional[DataLoader],
        fold: int,
        epoch: int
    ) -> Optional[dict[str, Metric]]:
        """Process one iteration of `DataLoader`"""
        env.stage, env.fold, env.epoch = stage, fold, epoch
        if loader is None:
            return None

        with WithNone() if stage == 'train' else torch.no_grad(), EpochProgress(stage, len(loader), fold, epoch) as progress:
            getattr(self.model_module, f'{stage}_epoch_begin')()

            for model in self.model_module.named_models.values():
                model.train() if stage == 'train' else model.eval()

            for step, batch in enumerate(loader, 1):
                step_res = self._do_step(stage, step, batch) or {}
                progress.update(**step_res)

                self._update_tensorboard(stage, self.model_module.named_metrics, step=epoch * len(loader) + step)
            self._update_tensorboard(stage, self.model_module.named_metrics, epoch=epoch)

            getattr(self.model_module, f'{stage}_epoch_end')()

            named_metrics = copy.deepcopy(self.model_module.named_metrics)
            for metric in self.model_module.named_metrics.values():
                metric.reset()
            return named_metrics

    def _do_step(
        self, 
        stage: str, 
        step: int, 
        batch,
    ) -> None:
        """Process one batch of data."""
        env.step = step
        with autocast():
            batch = self._batch_to_cuda(batch)
            return getattr(self.model_module, f'{stage}_step')(batch)

    def _save_ckpt(
        self, 
        fold: Optional[int] = None, 
        epoch: Optional[int] = None
    ) -> None:
        """Save checkpoints to path f'{env.exp_path}/ckpt/{fold}'.

        You can rewrite `yangdl.ModelModule.save_ckpt` function to decide exactly what to save.
        """
        ckpt_path = f'{env.exp_path}/ckpt'
        if fold is not None:
            ckpt_path = os.path.join(ckpt_path, str(fold))
        os.makedirs(ckpt_path, exist_ok=True)
        ckpt_name = os.path.join(ckpt_path, 'best.pt' if epoch is None else f'{epoch}.pt')

        self.model_module.save_ckpt(ckpt_name)

    def _load_ckpt(
        self, 
        fold: Optional[int] = None, 
        epoch: Optional[int] = None
    ) -> None:
        """Load checkpoints from path f'{env.exp_path}/ckpt/{fold}'.

        It is mainly used to load the best model during the test stage.
        """
        ckpt_path = f'{env.exp_path}/ckpt'
        if fold is not None:
            ckpt_path = os.path.join(ckpt_path, str(fold))
        ckpt_name = os.path.join(ckpt_path, 'best.pt' if epoch is None else f'{epoch}.pt')

        self.model_module.load_ckpt(ckpt_name)

    def _named_metrics2res(self, named_metrics: list[dict[str, Metric]]) -> dict:
        """Convert named_metrics to a combination of dict and list.

        Returns:
            res example: {'loss': {'loss': {'data': [0.1, 0.3], 'mean': 0.2, 'std': 0.01}}}.
        """
        res = defaultdict(lambda: defaultdict(lambda: {'data': [], 'mean': None, 'std': None}))
        for named_metric in named_metrics:
            for metric_name, metric in named_metric.items():
                for prop_name, prop in metric.data.items():
                    res[metric_name][prop_name]['data'].append(prop)

        for metric_dict in res.values():
            for prop_dict in metric_dict.values():
                arr = np.array(prop_dict['data'])
                prop_dict['mean'] = arr.mean(axis=0).tolist()
                prop_dict['std'] = arr.std(axis=0).tolist()

        return res
    
    def _simplify_res(self, res: dict) -> dict:
        """Remove res's 'data' and 'std', only keep 'mean'. Because people care more about the mean value of metrics."""
        res = copy.deepcopy(res)
        for _, stage_res in res.items():
            for metric_dict in stage_res.values():
                for prop_name, prop_dict in metric_dict.items():
                    metric_dict[prop_name] = prop_dict['mean']

        return res

    def _update_tensorboard(
        self,
        stage: str,
        named_metrics: dict[str, Metric],
        epoch: Optional[int] = None,
        step: Optional[int] = None
    ):
        """Log the scalar value of named_metrics to tensorboard."""
        if epoch == -1:
            return
        for metric in named_metrics.values():
            if epoch is not None and metric.freq[0] == 'epoch' and epoch % metric.freq[1] == 0:
                self.tensorboard_logger.log(metric, stage, epoch)
            if step is not None and metric.freq[0] == 'step' and step % metric.freq[1] == 0:
                self.tensorboard_logger.log(metric, stage, step)
    
    def _batch_to_cuda(self, batch):
        """Move cpu data in batch to gpu.

        TODO: Refactor this part.
        """
        if isinstance(batch, Tensor):
            batch = batch.cuda()
        elif isinstance(batch, dict):
            for key, val in batch.items():
                if isinstance(val, Tensor):
                    batch[key] = val.cuda()
        elif isinstance(batch, list):
            if isinstance(batch[0], (list, tuple)):
                batch = list(list(map(lambda x: x.cuda() if isinstance(x, Tensor) else x, item)) for item in batch)
            else:
                batch = [(item.cuda() if isinstance(item, Tensor) else item) for item in batch]
        else:
            raise Exception

        return batch

