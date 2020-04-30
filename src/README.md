This folder contains the core code for the project.

To generate the data used in the experiment use the command `python generate_solved_mip_instance.py --minutes 5`.  Note that this will take several days to complete.  

The python notebook `run_lin_ucb.ipynb` contains the experiements detailed in the report.  This trains LinUCB simulating an online enviornment using the offline data genereated with `generate_solved_mip_instance.py`.  The remained of the code is used to define the enviornment, interact with SCIP, and the data.  A summary of the files contained is provided below:
  - `data_loader.py` - This is used to get paths to MIP instances as well as their optimal objective values.
  - `enviornment.py` - a class that implements the core interatction with SCIP.  It implemenets the enviornemnt using a step() and reset(), similar to that of OpenAI gym.  
  - `reward.py`  - a class that implements the computation of several possible rewards which could be used for a reinforcement learning task.
  - `lin_ucb` - implements LinUCB with disjoint models.
  - `run_lin_ucb.ipynb` - the notebook which genereates the results presented in the report.  This notebook runs LinUCB on the data which was generated offline.   To generate the data used in the experiment run
  - `generate_solved_mip_instances.py` - this is used to generate the data for solving MIP instances over several seeds which was used in the experiments.
  - `bandit_utils` - a file which contains some utility functions used to sample rewards from the offline data which was generated.  
  - `utils.py` - a file which includes some general utility functions used in the core code.
 
