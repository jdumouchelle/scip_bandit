This folder contains the core code for the project.

To generate the data used in the experiment use the command `python generate_solved_mip_instance.py --minutes 5`.  Note that this will take several days to complete.  As the data is stored in `../solved_mip_results/` this command does not need to be run again in order to reproduce the experiments with LinUCB.  If attempting to reproduce the data, note that there will be differences as SCIPs results will vary based on machine.  

The python notebook `run_lin_ucb.ipynb` contains the experiments detailed in the report.  This trains LinUCB simulating an online environment using the offline data generated with `generate_solved_mip_instance.py`.  The remained of the code is used to define the environment, interact with SCIP, and the data.  A summary of the files contained is provided below:
  - `data_loader.py` - This is used to get paths to MIP instances as well as their optimal objective values.
  - `enviornment.py` - a class that implements the core interaction with SCIP.  It implements the environment using a step() and reset(), similar to that of OpenAI gym.  
  - `reward.py`  - a class that implements the computation of several possible rewards which could be used for a reinforcement learning task.
  - `lin_ucb` - implements LinUCB with disjoint models.
  - `run_lin_ucb.ipynb` - the notebook which generates the results presented in the report.  This notebook runs LinUCB on the data which was generated offline.   To generate the data used in the experiment run
  - `generate_solved_mip_instances.py` - this is used to generate the data for solving MIP instances over several seeds which was used in the experiments.
  - `bandit_utils` - a file which contains some utility functions used to sample rewards from the offline data which was generated.  
  - `utils.py` - a file which includes some general utility functions used in the core code.

