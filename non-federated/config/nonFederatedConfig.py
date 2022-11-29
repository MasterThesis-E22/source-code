from dataclasses import dataclass
from typing import Optional, Any, IO
import yaml
import os

@dataclass
class OptimizerConfig:
    name: str
    learning_rate: Optional[float] = None
    weight_decay: Optional[float] = None
    epsilon: Optional[float] = None
    alpha: Optional[float] = None
    momentum: Optional[float] = None


@dataclass
class DataConfig:
    batch_size: int
    oversampling: bool
    class_weights: bool


@dataclass
class ModelConfig:
    name: str


@dataclass
class TrainerConfig:
    epochs: int

@dataclass
class ExperimentConfig:
    project_name: str
    checkpoint_path: str
    metric_monitor: str
    metric_mode: str

@dataclass
class NonFederatedConfig:
    optimizer: OptimizerConfig
    data: DataConfig
    model: ModelConfig
    trainer: TrainerConfig
    experiment: ExperimentConfig

    def __post_init__(self):
        self.optimizer = OptimizerConfig(**self.optimizer)
        self.data = DataConfig(**self.data)
        self.model = ModelConfig(**self.model)
        self.trainer = TrainerConfig(**self.trainer)
        self.experiment = ExperimentConfig(**self.experiment)