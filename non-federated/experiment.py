import getopt
import sys
from typing import Dict, Any

import yaml

from config.nonFederatedConfig import NonFederatedConfig
from experiments.cifar10Experiment import Cifar10Experiment
from experiments.embryosExperiment import EmbryosExperiment
from experiments.experiment import Experiment
from experiments.mnistExperiment import MNISTExperiment


def read_yaml_file(path) -> NonFederatedConfig:
    with open(path) as yaml_file:
        content = yaml_file.read()
        contentDict = yaml.safe_load(content)
        config = NonFederatedConfig(**contentDict)
        return config

def main(argv):
    # Handle input config file
    configfile = ''
    try:
      opts, args = getopt.getopt(argv,"c:",["cfile="])
    except getopt.GetoptError:
      print('experiment.py -c <configfile>')
      sys.exit(2)
    for opt, arg in opts:
      if opt == '-h':
         print('experiment.py -c <configfile>')
         sys.exit()
      elif opt in ("-c", "--cfile"):
         configfile = arg

    # Start experiment based on config
    config = read_yaml_file(configfile)

    if config.experiment.project_name == "MNIST":
       experiment = MNISTExperiment(config)
    elif config.experiment.project_name == "Cifar10":
        experiment = Cifar10Experiment(config)
    elif config.experiment.project_name == "Embryos":
        experiment = EmbryosExperiment(config)
    else:
        raise NotImplementedError

    experiment.run()

if __name__ == "__main__":
   main(sys.argv[1:])
