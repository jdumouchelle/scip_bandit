import numpy as np 
import math
import os 
import sys
import pickle

from pyscipopt import Model

from data_loader import *
from reward import Reward
from utils import *



class Enviornment(object):

	def __init__(self, priority_or_freq, heuristics_to_run, time_limit=600):
		'''
			Constuctor for enviornment.
			Params:
				priority_or_freq - a string indicating the type of problem that should be focused on priority or frequency.
				heurisitcs_to_run - a list of the heuristics to run.
				time_limit - time limit for models in seconds.
		'''
		self.priority_or_freq = priority_or_freq
		self.time_limit = time_limit

		self.heuristics = ['actconsdiving' ,'bound' ,'clique' ,'coefdiving' ,'completesol' ,'conflictdiving' ,'crossover',
						   'dins' ,'distributiondiving' ,'dualval' ,'farkasdiving' ,'feaspump' ,'fixandinfer' ,'fracdiving' ,
						   'gins' ,'guideddiving' ,'zeroobj' ,'indicator' ,'intdiving' ,'intshifting' ,'linesearchdiving' ,
						   'localbranching' ,'locks' ,'lpface' ,'alns' ,'nlpdiving' ,'mutation' ,'multistart' ,'mpec' ,'objpscostdiving' ,
						   'octane' ,'ofins' ,'oneopt' ,'proximity' ,'pscostdiving' ,'randrounding' ,'rens' ,'reoptsols' ,'repair' ,'rins' ,
						   'rootsoldiving' ,'rounding' ,'shiftandpropagate' ,'shifting' ,'simplerounding' ,'subnlp' ,'trivial' ,'trivialnegation' ,
						   'trysol' ,'twoopt' ,'undercover' ,'vbounds' ,'veclendiving' ,'zirounding' ]

		self.heuristics_to_run = heuristics_to_run
		self.heuristics_to_disable = list(filter(lambda x: x not in self.heuristics_to_run, self.heuristics))

		self.state = None
		self.model = None

		return

	def seed_scip(self, seed):
		'''
			Sets seeds in SCIP.
			Params:
				seed - the seed to set seed params SCIP with.  
		'''
		self.model.setIntParam('randomization/randomseedshift', seed)
		self.model.setIntParam('randomization/lpseed', seed)
		self.model.setIntParam('randomization/permutationseed', seed)

		return


	def disable_heuristics(self):
		'''
			Disables all heurisitics except coefdiving, fracdiving, veclendiving
		'''
		for heur_to_disable in self.heuristics_to_disable:
			self.model.setIntParam('heuristics/' + heur_to_disable + '/freq', -1)
			self.model.setIntParam('heuristics/' + heur_to_disable + '/priority', -100000)

		return


	def set_time_limit(self, time_limit):
		'''
			Sets the time limit of the scip model.
			Params:
				time_limit - the time limit in seconds.
		'''

		self.model.setRealParam('limits/time', time_limit)

		return


	def set_priority(self, action):
		'''
			Sets priority of all heursitics.  
			Params:
				action - a dictionary of actions.
		'''

		for i in range(len(self.heuristics_to_run)):

			heur_to_run = self.heuristics_to_run[i]
			action_value = action[heur_to_run]
			self.model.setIntParam('heuristics/' + heur_to_run + '/priority', action_value) # set priority
			print('Setting: heuristics/' + heur_to_run + '/priority', action_value)

		return



	def set_freq(self, action):
		'''
			Sets frequency of all heursitics.  
			Params:
				action - a dictionary of actions.
		'''

		for i in range(len(self.heuristics_to_run)):

			heur_to_run = self.heuristics_to_run[i]
			action_value = action[heur_to_run]
			self.model.setIntParam('heuristics/' + heur_to_run + '/freq', action_value) # set frequency
			self.model.setIntParam('heuristics/' + heur_to_run + '/priority', 1) # set priority
			print('heuristics/' + heur_to_run + '/freq', action_value)

		return



	def get_state(self):
		'''
			Gets the state of the model.  The state is simply based 
			on the number of variables/constrains and the type of the 
			variables as these are avaliable without any knowledge of 
			the solving process.  
		'''

		# get number of constraints and variables
		num_constrains = self.model.getNConss()
		num_variables = self.model.getNVars()
		
		# get number of each variable type
		num_binary_variables = 0
		num_integer_variables = 0
		num_continous_variables = 0
		variables = self.model.getVars()

		for variable in variables:
			if variable.vtype() == 'BINARY':
				num_binary_variables += 1
			elif variable.vtype() == 'INTEGER':
				num_integer_variables += 1
			else:
				num_continous_variables += 1

		state = {
			'num_constrains' : num_constrains,
			'num_variables' : num_variables,
			'num_binary_variables' : num_binary_variables, 
			'num_integer_variables' : num_integer_variables,
			'num_continous_variables' : num_continous_variables
		}

		return state



	def reset(self, path, seed=0):
		'''
			Loads a random instance and returns state given by the instance.  
			Params:
				path - path to write results to. 
				seed - seed for seeding SCIP params.
			Returns:
				a dictionary continaing all state information.  
		'''
		
		self.model = Model() 
		self.model.readProblem(path)

		self.seed_scip(seed)

		self.state = self.get_state()

		return self.state 


	def step(self, action, write_path=None, objective_value=None):
		'''
			Takes a step in the enviornment.
			Params:
				action - the action to be taken
				write_path - path to write_statistics to
				objective_value - the objective value of the 
			Returns:
				an object of the rewards
		'''

		# disable heuristics that are not part of the MDP
		self.disable_heuristics()

		# set priorities or frequencies
		if self.priority_or_freq == 'priority':
			self.set_priority(action)
		else:
			self.set_freq(action)

		# compute primal bound at each time step
		primal_bounds = []
		dual_bounds = []
		for time_step in range(self.time_limit):
			self.set_time_limit(time_step)
			self.model.optimize()
			primal_bounds.append(self.model.getPrimalbound())
			dual_bounds.append(self.model.getDualbound())

		# write model statistics
		if write_path is not None:
			self.model.writeStatistics(write_path)

		# get rewards
		reward = Reward(self.model, primal_bounds, dual_bounds, objective_value)

		return reward.get_reward_as_dict()





