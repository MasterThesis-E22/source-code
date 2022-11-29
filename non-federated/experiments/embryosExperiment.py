from config.nonFederatedConfig import NonFederatedConfig
from experiments.experiment import Experiment
from models.embryosLightning import EmbryosLightning


class EmbryosExperiment(Experiment):
    def __init__(self, config: NonFederatedConfig):
        super().__init__(config)
        self.model = EmbryosLightning(config)

    def run(self):
        self.trainer.fit(self.model)

        best_model_path = self.checkpoint_callback.best_model_path
        best_model = EmbryosLightning.load_from_checkpoint(best_model_path)
        self.trainer.test(model=best_model)

    def test(self, path: str):
        best_model = EmbryosLightning.load_from_checkpoint(path)
        self.trainer.test(model=best_model)

