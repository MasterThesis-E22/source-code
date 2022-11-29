from dataclasses import asdict

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config.nonFederatedConfig import NonFederatedConfig


class Experiment:
    def __init__(self, config: NonFederatedConfig):
        self.config = config

        # Initialize logger
        self.wandb_logger = WandbLogger(project=self.config.experiment.project_name,
                                        name=f"{self.config.model.name}_{self.config.optimizer.name}",
                                        log_model="all", save_dir="wandb/", group="non-federated")
        self.wandb_logger.experiment.config.update({k: str(v) for k, v in asdict(self.config.optimizer).items()},
                                                   allow_val_change=True)
        self.wandb_logger.experiment.config.update({k: str(v) for k, v in asdict(self.config.model).items()},
                                                   allow_val_change=True)
        self.wandb_logger.experiment.config.update({k: str(v) for k, v in asdict(self.config.data).items()},
                                                   allow_val_change=True)
        self.wandb_logger.experiment.config.update({k: str(v) for k, v in asdict(self.config.trainer).items()},
                                                   allow_val_change=True)
        # Initialize callbacks
        self.checkpoint_callback = ModelCheckpoint(dirpath=self.config.experiment.checkpoint_path,
                                                   save_top_k=2,
                                                   monitor=self.config.experiment.metric_monitor,
                                                   mode=self.config.experiment.metric_mode,
                                                   filename="{epoch:03d}",
                                                   save_last=True)

        # Initialize trainer
        self.trainer = Trainer(
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            max_epochs=self.config.trainer.epochs,
            callbacks=[self.checkpoint_callback,
                       ],
            enable_progress_bar=False,
            logger=self.wandb_logger,
        )

    def run(self):
        raise NotImplementedError()

    def test(self, path: str):
        raise NotImplementedError()

