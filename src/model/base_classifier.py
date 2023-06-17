import abc
from typing import Any

import torch
import torch.nn.functional as F
from pl_bolts.models import VAE
from pytorch_lightning import LightningModule
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torchmetrics.functional import accuracy
from torchmetrics.functional.classification import multiclass_calibration_error


class BaseClassifier(LightningModule, abc.ABC):
    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch: Any, stage=None):
        x, y = batch
        logits = self(x)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        ece = multiclass_calibration_error(F.softmax(logits, dim=1), y, num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_ece", ece, prog_bar=True)

    def validation_step(self, batch: Any, batch_idx: int):
        self.evaluate(batch, "val")

    def test_step(self, batch: Any, batch_idx: int):
        self.evaluate(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        return {"optimizer": optimizer}


class SimCLRClassifier(BaseClassifier):
    def __init__(self, model: VAE, lr: float = 0.05, batch_size: int = 256):
        super().__init__()
        self.save_hyperparameters(ignore=["model"])

        self.model = model

        # the variational autoencoder outputs a 2048-dim representation and CIFAR-10 has 10 classes
        self.classifier = nn.Linear(2048, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.model(x)
        out = self.classifier(z)
        return F.log_softmax(out, dim=1)

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        (x1, x2, x3), y = batch
        logits = self.forward(x3)
        loss = F.nll_loss(logits, y)
        self.log("train_loss", loss)
        return loss

    def evaluate(self, batch: Any, stage=None):
        (x1, x2, x3), y = batch
        logits = self.forward(x3)
        loss = F.nll_loss(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y, task="multiclass", num_classes=10)
        ece = multiclass_calibration_error(F.softmax(logits, dim=1), y, num_classes=10)

        if stage:
            self.log(f"{stage}_loss", loss, prog_bar=True)
            self.log(f"{stage}_acc", acc, prog_bar=True)
            self.log(f"{stage}_ece", ece, prog_bar=True)
