"""
A simple example to demonstrate how to use `yangdl` to complete kaggle competition (Digit Recognizer)[https://www.kaggle.com/c/digit-recognizer].

Follow the steps below to run this code (you need to have kaggle api first, see https://github.com/Kaggle/kaggle-api for details).
1. kaggle competitions download -c digit-recognizer
2. unzip digit-recognizer.zip -d digit-recognizer
3. mv yangdl/examples/digit-recognizer.py digit-recognizer
4. cd digit-recognizer
5. python digit-recognizer.py

Then you will find result in `./res/baseline` directory.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.nn import functional as F
from torchvision.models.resnet import resnet18
from torchvision.transforms import transforms as T

import yangdl as yd


EXP_PATH = "./res/baseline"
yd.env.exp_path = EXP_PATH  # path to save experimental results
yd.env.seed = 1  # set seed for reproduction

BATCH_SIZE = 256
OUT_PATH = f"{EXP_PATH}/preds"
os.makedirs(OUT_PATH, exist_ok=True)


# 1. write your own ModelModule with train_step, val_step, test_step, predict_step(don't need all)
class MyModelModule(yd.ModelModule):
    def __init__(self):
        super().__init__()

        self.criterion = nn.CrossEntropyLoss()

        # used to calculate experimental results
        self.loss = yd.ValueMetric()
        self.metric = yd.ClsMetric(num_classes=10)

        self.scaler = GradScaler()

    def __iter__(self):
        while True:
            # initialize new model for new fold
            self.model = resnet18(num_classes=10)

            # initialize model's optimizer
            self.optimizer = AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=1e-4)

            # use to save preds in predict_step
            self.preds = []

            yield

    # input batch and return dict for showing in the progress bar
    def train_step(self, batch):
        loss = self._step(batch)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

        return {"loss": loss, "acc": self.metric.acc}

    def val_step(self, batch):
        loss = self._step(batch)
        return {"loss": loss, "acc": self.metric.acc}

    # predict labels for each batch
    def predict_step(self, batch):
        x = batch["image"]

        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        self.preds.append(preds.cpu().numpy())

    # usually log epoch result hear
    def train_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            auc=self.metric.auc,
            acc=self.metric.acc,
            f1_score=self.metric.f1_score,
        )

    def val_epoch_end(self):
        yd.logger.log_props(
            loss=self.loss.val,
            auc=self.metric.auc,
            acc=self.metric.acc,
            f1_score=self.metric.f1_score,
        )

    # save preds to '{EXP_PATH}/pred/{fold}.csv'
    def predict_epoch_end(self):
        pred_filename = f"{OUT_PATH}/{yd.env.fold}.csv"
        preds = np.concatenate(self.preds)
        pd.DataFrame({"ImageID": range(1, 28001), "Label": preds}).to_csv(
            pred_filename, index=None
        )

        print(f"predict fold {yd.env.fold} finished!")

    def _step(self, batch):
        x, y = batch["image"], batch["label"]

        with autocast():
            logits = self.model(x)
            loss = self.criterion(logits, y)
            probs = F.softmax(logits, dim=1)

        self.loss.update(loss, x.shape[0])
        self.metric.update(probs, y)

        return loss


# 2.1 write your own Dataset for DataModule
class MyDataset(Dataset):
    def __init__(self, images, labels=None):
        super().__init__()

        self.images = images[..., None].repeat(3, axis=-1).astype("float32")
        self.labels = labels

        self.transform = T.Compose(
            [
                T.ToTensor(),
                T.Resize((56, 56), antialias=True),
                T.Normalize(0.1310, 0.3085),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.transform(self.images[idx])
        if self.labels is not None:
            label = self.labels[idx]
            return {"image": image, "label": label}
        else:
            return {"image": image}


# 2.2 write your own DataModule for generate DataLoader
class MyDataModule(yd.DataModule):
    def __init__(self):
        super().__init__()

        train = pd.read_csv("./train.csv")
        self.train_images = train.iloc[:, 1:].values.reshape(-1, 28, 28)
        self.train_labels = train.iloc[:, 0].values

        test = pd.read_csv("./test.csv")
        self.test_images = test.values.reshape(-1, 28, 28)

        # 5-fold cross validation
        skf = StratifiedKFold(5, shuffle=True, random_state=1)
        self.train_idxs, self.val_idxs = [], []
        for train_idx, val_idx in skf.split(self.train_images, self.train_labels):
            self.train_idxs.append(train_idx)
            self.val_idxs.append(val_idx)

    # use to generate different DataLoader for different fold
    def train_loader(self):
        for train_idx in self.train_idxs:
            dataset = MyDataset(
                self.train_images[train_idx], self.train_labels[train_idx]
            )
            yield DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                num_workers=4,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            )

    def val_loader(self):
        for val_idx in self.val_idxs:
            dataset = MyDataset(self.train_images[val_idx], self.train_labels[val_idx])
            yield DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                num_workers=4,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )

    def predict_loader(self):
        while True:
            dataset = MyDataset(self.test_images)
            yield DataLoader(
                dataset,
                batch_size=BATCH_SIZE,
                num_workers=4,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            )


if __name__ == "__main__":
    # use ModelModule, DataModule to initialize TaskModule
    task_module = yd.TaskModule(
        early_stop_params={
            "monitor": {"loss.val": "min"},
            "patience": 5,
            "min_stop_epoch": 5,
            "max_stop_epoch": 25,
        },
        benchmark=True,
        model_module=MyModelModule(),
        data_module=MyDataModule(),
    )

    # begin your experiment!
    task_module.do()
