import numpy as np
import torch
import torch.nn as nn
import wandb
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, Precision, Recall, AUC, AUROC, F1Score, ConfusionMatrix
from torchmetrics.classification import MulticlassAUROC, MulticlassAccuracy, MulticlassPrecision, MulticlassRecall, \
    MulticlassF1Score
from torchvision import transforms

from config.nonFederatedConfig import NonFederatedConfig
from dataset.mnist import MNIST
from models.models import MNISTLowGPUCNN


class View(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.view(-1, 1, 28, 28)


class MNISTLightning(LightningModule):
    def __init__(self, config: NonFederatedConfig):
        super().__init__()
        self.config = config

        self.num_classes = 10
        self.dims = (1, 28, 28)
        channels, width, height = self.dims

        self.model = MNISTLowGPUCNN()

        self.val_accuracy = MulticlassAccuracy(average='macro', num_classes=self.num_classes)
        self.test_accuracy = MulticlassAccuracy(average='macro', num_classes=self.num_classes)

        self.val_precision = MulticlassPrecision(average='macro', num_classes=self.num_classes)
        self.test_precision = MulticlassPrecision(average='macro', num_classes=self.num_classes)

        self.val_recall = MulticlassRecall(average='macro', num_classes=self.num_classes)
        self.test_recall = MulticlassRecall(average='macro', num_classes=self.num_classes)

        self.val_f1 = MulticlassF1Score(average='macro', num_classes=self.num_classes)
        self.test_f1 = MulticlassF1Score(average='macro', num_classes=self.num_classes)

        self.val_auroc = MulticlassAUROC(num_classes=self.num_classes)
        self.test_auroc = MulticlassAUROC(num_classes=self.num_classes)

        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return F.log_softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.log('train/loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.val_accuracy.update(logits, y)
        self.val_precision.update(logits, y)
        self.val_recall.update(logits, y)
        self.val_f1.update(logits, y)
        self.val_auroc.update(logits, y)

        self.log("val/loss", loss, on_step=False, on_epoch=True)

    def validation_epoch_end(self, outputs):
        self.log("val/accuracy", self.val_accuracy.compute())
        self.log("val/precision", self.val_precision.compute())
        self.log("val/recall", self.val_recall.compute())
        self.log("val/f1", self.val_f1.compute())
        self.log("val/auroc", self.val_auroc.compute())

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = F.cross_entropy(logits, y)

        self.test_accuracy.update(logits, y)
        self.test_precision.update(logits, y)
        self.test_recall.update(logits, y)
        self.test_f1.update(logits, y)
        self.test_auroc.update(logits, y)

        self.log("test/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/accuracy", self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/precision", self.test_precision, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/recall", self.test_recall, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test/auroc", self.test_auroc, prog_bar=True, on_step=False, on_epoch=True)
        return {"logits": logits, "labels": y}

    def test_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        labels = labels.cpu()
        logits = logits.cpu()
        preds  = torch.argmax(logits, dim=1)
        class_names = range(0, 10)

        self.logger.experiment.log(
            {
                "roc": wandb.plot.roc_curve(y_true=labels.numpy(), y_probas=logits.numpy(), labels=class_names)
            })

        self.logger.experiment.log(
            {
                "pr": wandb.plot.pr_curve(y_true=labels.numpy(), y_probas=logits.numpy(), labels=class_names)
            })

        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(preds=preds.numpy(),
                                                    y_true=labels.numpy(),
                                                    class_names=class_names)
            })

        self.logger.experiment.log(
            {
                "conf-sklearn": wandb.sklearn.plot_confusion_matrix(labels.numpy(), preds.numpy(), class_names)
            })

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.config.optimizer.learning_rate, weight_decay=self.config.optimizer.weight_decay)
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        mnist = MNIST()

        self.train_data = mnist.get_train_dataset()
        self.valid_data = mnist.get_valid_dataset()
        self.test_data = mnist.get_test_dataset()

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config.data.batch_size, shuffle=False, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.config.data.batch_size, shuffle=False, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.config.data.batch_size, shuffle=False, num_workers=16)
