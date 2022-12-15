# Description
Non federated experiments

# Structure
.  
├── config (experiment configuration to run)  
├── dataset (the datasets used)  
├── experiments (experiment definitions)  
└── models (lightning modules and CNNs)

# Reproducibility
- Create a virtual python environment: python3 -m venv venv
- Activate the environment: . venv/bin/activate
- Install torch and torchvision (url depends on CUDA version): pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116
- Install other requirement defined requirements.txt: pip install -r requirements.txt
- Experiments can now be run 
  
