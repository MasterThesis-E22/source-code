# Description
Non federated experiments

# Structure
.
├── config (experiment configuration to run)
├── dataset (the datasets used)
├── experiments (experiment definitions)
└── models (lightning modules and CNNs)

# Reproducibility
There is a requirement.txt which can be used to create python virtual environment. The requirement.txt is created with pip freeze from a working environment. There is a note regarding torch and torchvision. These can be acquired from the --extra-index-url https://download.pytorch.org/whl/cu116 depending on the cuda version on the machine. 
  
