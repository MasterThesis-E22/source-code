from experiments.embryosExperiment import EmbryosExperiment
import yaml

from config.nonFederatedConfig import NonFederatedConfig

def read_yaml_file(path) -> NonFederatedConfig:
    with open(path) as yaml_file:
        content = yaml_file.read()
        contentDict = yaml.safe_load(content)
        config = NonFederatedConfig(**contentDict)
        return config

if __name__ == "__main__":
    config = read_yaml_file("config/debug.yaml")
    experiment = EmbryosExperiment(config)
    experiment.test("/home/maddox/plato-git/plato/experiments/sync/models/fedavg/embryos/FedAvg-LR1e4-C23/checkpoint_lowgpuEmbryosNew_95.pth")