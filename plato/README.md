# Description
The federated experiments used in the thesis.  

# Structure
.  
├── experiments (all experiment configurations)  
└── plato (source code for the plato framework)  

# Plato modifications
Modifications have been made to the existing plato framework. The changes that have been made is listed below:
- Added support for additional data sources. Pre-partitioned non-iid versions of Cifar-10 and MNIST, they can be generated with the data project. Furthermore, the Embryos data source is added.
- Added support for a binary classification problem
- Modification to the validation flow. Originally plato had to ways  of performing validation. Either on the client after it had modified its local model or on the server when the global model has been updated. Now doing it on the client we do it before the local model is updated, to get a representation of how the client performs on the global model and not the modified global model.
- Doing validation on the server we track the best validation round.
- Added a final test, where the best model from the best validation round is used. 
- Added a final validation, so instead of the server doing validation each round, we in the end process all rounds of global models and perform validation. 
- Plato originally only logs performance metrics to a .csv file, we added integration to WANDB. 
- Added more performance metrics to be logged. Plato originally only logged accuracy.
- Modification to staleness simulation in asynchronous mode: added possibility of adding delay, that is truly random each round. Furthermore, added possibility of simulating staleness with random number genertions. 

# How to run
- Create a virtual python environment with the defined requirement.txt. Note you need torch and torchvision as well. Here you need to install a version depending on your GPU, GPU driver and CUDA version. For instance we utilize CUDA 11.6 and install torch and  torchvision with the following command: "pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu116"
- Now you can run the experiment defined in the experiments folder. There are two folders on for asynchronous experiments and for synchronous. Each folder has a base that can be used for different configurations. 
- Navigating to a folder, an experiment can then be run: "python sync_base.py -c {filename}.yml" where filename is the desired configuration. An example is "python sync_base.py -c mnist/mnist_test.yml" 
- We have run the experiments on a remote server, and it is has therefore been necessary to queue multiple experiments. 
- This can be done with one of shell files. We run them with nohup so we are able to shut down our workstations: "nohup run_benchmark_experiments.sh &"
