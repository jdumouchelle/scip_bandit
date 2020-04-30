# scip_bandit
This project explores the use of contextual bandits for learning the frequency of primal heuristics in Mixed Integer Programming (MIP).  The MIP solver used was SCIP and information on how to obtain SCIP can be found below.  The data is sourced from MIPLIB 2010 and 2017 with instructions on downloading and storing found in the `data/README.md` file.  The report can be found in `report.pdf`.  The code for defining the enviornment, generating the MIP solving data, and running experiements can all be found in `src`.  The solved MIP data is located in `solved_mip_results/`.  Requirements for other packages used can be found in `requirements.txt`.

Installing SCIP:
 - Obtain SCIP 6.0.2 here: https://scip.zib.de/index.php#download 
 - Once installed, follow the instructions here https://github.com/SCIP-Interfaces/PySCIPOpt/blob/master/INSTALL.md to set up pyscipopt.
