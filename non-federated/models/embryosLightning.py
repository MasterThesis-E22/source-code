import numpy as np
import torch
import torch.nn as nn
import wandb.sklearn.plot
from pytorch_lightning import LightningModule
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryPrecision, BinaryRecall, BinaryAUROC, BinaryF1Score, BinaryStatScores
from torchvision import transforms

from config.nonFederatedConfig import NonFederatedConfig
from dataset.embroys import Embryos
from models.models import EmbryosLowGPUCNN, UpdatedEmbryosLowGPUCNN, MobileNetV2


class View(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(input):
        return input.view(-1, 1, 250, 250)


class EmbryosLightning(LightningModule):
    def __init__(self, config: NonFederatedConfig):
        super().__init__()
        self.test_data = None
        self.valid_data = None
        self.train_data = None
        self.example_input_array = torch.rand(5, 3, 250, 250)

        self.config = config

        self.num_classes = 2
        self.dims = (3, 250, 250)

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )
        if self.config.model.name == "EmbryosLowGPUCNN":
            self.model = EmbryosLowGPUCNN(1)
        elif self.config.model.name == "UpdatedEmbryosLowGPUCNN":
            self.model = UpdatedEmbryosLowGPUCNN(1)
        elif self.config.model.name == "MobileNetV2":
            self.model = MobileNetV2(self.num_classes, 0.55)

        self.train_accuracy = BinaryAccuracy()
        self.val_accuracy = BinaryAccuracy()
        self.test_accuracy = BinaryAccuracy()

        self.val_precision = BinaryPrecision()
        self.test_precision = BinaryPrecision()

        self.val_recall = BinaryRecall()
        self.test_recall = BinaryRecall()

        self.val_f1 = BinaryF1Score()
        self.test_f1 = BinaryF1Score()

        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

        self.train_statscore = BinaryStatScores()
        self.val_statscore = BinaryStatScores()

        self.save_hyperparameters()

    def forward(self, x):
        x = self.model(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = logits.squeeze(dim=1)

        if(self.config.data.class_weights):
            pos_weight = torch.FloatTensor([2.1]).to("cuda:0")
        else:
            pos_weight = None
        loss = F.binary_cross_entropy_with_logits(logits, y.float(), pos_weight=pos_weight)

        self.train_statscore.update(logits, y)
        self.train_accuracy.update(logits, y)
        self.log('train/loss', loss, on_step=False, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        self.log('train/accuracy', self.train_accuracy.compute())

        stat_scores = self.train_statscore.compute()
        self.log('train/tp', stat_scores[0].item())
        self.log('train/fp', stat_scores[1].item())
        self.log('train/tn', stat_scores[2].item())
        self.log('train/fn', stat_scores[3].item())

        self.train_accuracy.reset()
        self.train_statscore.reset()

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = logits.squeeze(dim=1)
        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        self.val_accuracy.update(logits, y)
        self.val_precision.update(logits, y)
        self.val_recall.update(logits, y)
        self.val_f1.update(logits, y)
        self.val_auroc.update(logits, y)
        self.val_statscore.update(logits, y)

        self.log("val/loss", loss)

    def validation_epoch_end(self, outputs):
        self.log("val/accuracy", self.val_accuracy.compute())
        self.log("val/precision", self.val_precision.compute())
        self.log("val/recall", self.val_recall.compute())
        self.log("val/f1", self.val_f1.compute())
        self.log("val/auroc", self.val_auroc.compute())

        stat_scores = self.val_statscore.compute()
        self.log('val/tp', stat_scores[0].item())
        self.log('val/fp', stat_scores[1].item())
        self.log('val/tn', stat_scores[2].item())
        self.log('val/fn', stat_scores[3].item())

        self.val_accuracy.reset()
        self.val_precision.reset()
        self.val_recall.reset()
        self.val_f1.reset()
        self.val_auroc.reset()
        self.val_statscore.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        logits = logits.squeeze(dim=1)

        loss = F.binary_cross_entropy_with_logits(logits, y.float())

        self.test_accuracy.update(logits, y)
        self.test_precision.update(logits, y)
        self.test_recall.update(logits, y)
        self.test_f1.update(logits, y)
        self.test_auroc.update(logits, y)

        # Calling self.log will surface up scalars for you in TensorBoard
        self.log("test/loss", loss)

        return {"logits": logits, "labels": y}

    def test_epoch_end(self, outputs):
        labels = torch.cat([x["labels"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])
        labels = labels.cpu()
        logits = logits.cpu()
        probs = torch.sigmoid(logits)
        both_probs = np.vstack((1 - probs.numpy(), probs)).T
        preds = (logits > 0).long()
        class_names = ["No", "Yes"]

        self.logger.experiment.log(
            {
                "roc": wandb.plot.roc_curve(labels.numpy(), both_probs, class_names)
            })

        self.logger.experiment.log(
            {
                "pr": wandb.plot.pr_curve(y_true=labels.numpy(), y_probas=both_probs, labels=class_names)
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

        self.log("test/accuracy", self.test_accuracy.compute())
        self.log("test/precision", self.test_precision.compute())
        self.log("test/recall", self.test_recall.compute())
        self.log("test/f1", self.test_f1.compute())
        self.log("test/auroc", self.test_auroc.compute())

        self.test_accuracy.reset()
        self.test_precision.reset()
        self.test_recall.reset()
        self.test_f1.reset()
        self.test_auroc.reset()

    def configure_optimizers(self):
        if self.config.optimizer.name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(),
                                        lr=self.config.optimizer.learning_rate,
                                        weight_decay=self.config.optimizer.weight_decay,
                                        momentum=self.config.optimizer.momentum)
        elif self.config.optimizer.name == "Adam":
            optimizer = torch.optim.Adam(self.parameters(),
                                         lr=self.config.optimizer.learning_rate,
                                         weight_decay=self.config.optimizer.weight_decay,
                                         eps=self.config.optimizer.epsilon)
        elif self.config.optimizer.name == "RMSProp":
            optimizer = torch.optim.RMSprop(self.parameters(),
                                            lr=self.config.optimizer.learning_rate,
                                            weight_decay=self.config.optimizer.weight_decay,
                                            eps=self.config.optimizer.epsilon,
                                            momentum=self.config.optimizer.momentum,
                                            alpha=self.config.optimizer.alpha)
        else:
            raise NotImplementedError
        return optimizer

    ####################
    # DATA RELATED HOOKS
    ####################

    def setup(self, stage=None):
        no_channels = 1
        if self.config.model.name == "MobileNetV2":
            no_channels = 3
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            embryos = Embryos(1,
                              oversampling=self.config.data.oversampling,
                              no_channels=no_channels)
            self.train_data = embryos.get_train_dataset()
            self.valid_data = embryos.get_valid_dataset()

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            embryos = Embryos(1, no_channels=no_channels)
            self.test_data = embryos.get_test_dataset()

    def train_dataloader(self):
        return DataLoader(self.train_data, batch_size=self.config.data.batch_size, shuffle=False, num_workers=16)

    def val_dataloader(self):
        return DataLoader(self.valid_data, batch_size=self.config.data.batch_size, shuffle=False, num_workers=16)

    def test_dataloader(self):
        return DataLoader(self.test_data, batch_size=self.config.data.batch_size, shuffle=False, num_workers=16)
