from rich.progress import (
    Progress,
    BarColumn,
    MofNCompleteColumn,
)


__all__ = [
    'EpochProgress',
]


class EpochProgress(Progress):
    """Display the experimental process in the form of a progress bar."""
    def __init__(self, stage: str, total: int, fold: int, epoch: int):
        super().__init__(transient=True)

        self.columns = (
            '{task.description}: fold: {task.fields[fold]}, epoch: {task.fields[epoch]}',
            BarColumn(),
            MofNCompleteColumn(),
        )
        self.keys = set()
        self.color = {'train': 'red', 'val': 'blue', 'test': 'green', 'predict': 'pink'}[stage]
        self.task_id = self.add_task(description=f'[{self.color}]{stage}[/{self.color}]', total=total, fold=fold, epoch=epoch)

    def update(self, **kwargs):
        """Display the `**kwargs` behind the progress bar."""
        for key in kwargs:
            if key not in self.keys:
                self.keys.add(key)
                self.columns = (*self.columns, f'{key}: [{self.color}]{{task.fields[{key}]:.3f}}[/{self.color}]')

        super().update(self.task_id, **kwargs)
        super().advance(self.task_id)

