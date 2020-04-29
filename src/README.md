This folder contains all the core code for the enviornemnt, data interation, and the contextual bandit. The files contained are:
  - data_loader.py - This is used to get paths to MIP instances as well as their optimal objective values.
  - enviornment.py - a class that implements the core interatction with SCIP.  It implemenets the enviornemnt using a step() and reset(), similar to that of OpenAI gym.  
  - reward.py  - a class that implements the computation of several possible rewards which could be used for a reinforcement learning task.
  - lin_ucb - implements LinUCB with disjoint models.
  - run_lin_ucb.ipynb - the notebook which genereates the results presented in the report.  This notebook runs LinUCB on the data which was generated offline. 
  - generate_solved_mip_instances.py - this is used to generate the data for solving MIP instances over several seeds which was used in the experiments.
  - bandit_utils - a file which contains some utility functions used to sample rewards from the offline data which was generated.  
  - utils.py - a file which includes some general utility functions used in the core code.
 
