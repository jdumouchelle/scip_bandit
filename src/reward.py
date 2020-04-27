import numpy as np

class Reward(object):

	def __init__(self, model, primal_bounds=None, dual_bounds=None, objective_value=None):
		'''
			Constructor for reward class.  This class simply compute and stores several rewards which can 
			be used in training the RL algorithm.  
		'''


		self.time_to_solve = model.getTotalTime()
		self.number_of_nodes = model.getNNodes()
		self.primal_dual_gap = model.getGap()
		self.LP_iterations = model.getNLPIterations()

		self.primal_bounds = primal_bounds
		self.primal_gaps = None
		self.primal_integral = None

		self.dual_bounds = dual_bounds
		
		if objective_value is not None:
			self.set_primal_integral(objective_value)

		
	def set_primal_integral(self, opt):
		'''
			Setter for the primal_integral.  Also sets primal_bounds and primal_gaps.
		'''
		def compute_primal_gaps(primal_bounds, opt):
			'''
				Computes primal gaps for each time step.
				Params:
				primal_bounds - a list of the primal_bounds at each time step.
				opt - the optimal solution to the MIP.
			'''
			

			primal_gaps = []
			for primal_bound in primal_bounds:
				
				if primal_bound == opt:
					primal_gaps.append(0)

				elif primal_bound * opt < 0:
					primal_gaps.append(1)
					
				else:
					primal_gaps.append(np.abs(primal_bound - opt) / np.max([np.abs(primal_bound), np.abs(opt)]))

			return primal_gaps

		def compute_primal_integral(primal_gaps):
			'''
				Computes the primal integral for the instances.  Since we are simplifying this a bit to have the
				incumbent at time t, for t = 1,2,...,max_time, the primal integral can simply be computed by the 
				sum.  
				Params:
					primal_gaps - the list of primal_gaps at each step.  
			'''
			primal_integral = np.sum(primal_gaps)

			return primal_integral

		self.primal_gaps = compute_primal_gaps(self.primal_bounds, opt)
		self.primal_integral = compute_primal_integral(self.primal_gaps)

		return


	def get_reward_as_dict(self):
		'''
			Stores all the rewards in a dictionary and returns.  
		'''
		reward_dict = {
			'time_to_solve' : self.time_to_solve,
			'number_of_nodes' : self.number_of_nodes,
			'primal_dual_gap' : self.primal_dual_gap,
			'LP_iterations' : self.LP_iterations,
			'primal_bounds' : self.primal_bounds,
			'primal_gaps' : self.primal_gaps,
			'primal_integral' : self.primal_integral,
			'dual_bounds' : self.dual_bounds,
		}

		return reward_dict
	
