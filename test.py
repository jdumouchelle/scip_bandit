import numpy as np 
import math
import os 
import sys


from data_loader import load_miplib, load_generated
from pyscipopt import Model



class Enviornment(object):

	def __init__(self, priority_or_freq, heuristics_to_run):
		'''
		'''

		self.priority_or_freq = priority_or_freq

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

	def seed(self, seed):
		self.seed = seed

		return


	def disable_heuristics(self):
		'''
			Disables all heurisitics except coefdiving, fracdiving, veclendiving
		'''
		for heur_to_disable in self.heuristics_to_disable:
			self.model.setIntParam('heuristics/' + heur_to_disable + '/freqofs', 65534)
			self.model.setIntParam('heuristics/' + heur_to_disable + '/priority', -100000)

		return


	def set_priority(self, action):
		'''
			Sets priorities for heurisitics coefdiving, fracdiving, veclendiving 
			according to an action
		'''

		for i in range(len(self.heuristics_to_run)):

			heur_to_run = self.heuristics_to_run[i]
			action_value = action[heur_to_run]
			self.model.setIntParam('heuristics/' + heur_to_run + '/priority', action_value) # set priority
			print('Setting: heuristics/' + heur_to_run + '/priority', action_value)

		return



	def set_freqofs(self, action):
		'''
			Sets frequency of coefdiving, fracdiving, veclendiving.
		'''

		for i in range(len(self.heuristics_to_run)):

			heur_to_run = self.heuristics_to_run[i]
			action_value = action[heur_to_run]
			self.model.setIntParam('heuristics/' + heur_to_run + '/freqofs', action_value) # set frequency
			print('heuristics/' + heur_to_run + '/freqofs', action_value)

		return



	def get_state(self):
		'''
			Gets the state of the model.  
		'''

		return 0


	def get_reward(self):
		'''
		'''
		
		reward_ts = self.model.getSolvingTime()
		reward_nn = self.model.getNNodes()

		reward = {'time_to_solve' : reward_ts,
				  'number_of_nodes' : reward_nn}

		return reward

	def reset(self, path, seed=0):
		'''
			Loads a random instance and returns state given by the instance.  
		'''
		self.model = Model() 
		self.model.readProblem(path)
		self.state = self.get_state()

		return self.state 


	def step(self, action):

		self.disable_heuristics()

		# set priorities
		if self.priority_or_freq == 'priority':
			self.set_priority(action)
		else:
			self.set_freqofs(action)
		

		self.model.optimize()

		# get reward
		reward = self.get_reward()

		return reward




def get_scip_default(heur, priority_or_freq):

	if heur == 'coefdiving' and priority_or_freq == 'priority':
		return -1001000

	elif heur == 'fracdiving' and priority_or_freq == 'priority':
		return -1003000

	elif heur == 'veclendiving' and priority_or_freq == 'priority':
		return -1003100

	elif heur == 'coefdiving' and priority_or_freq == 'freqofs':
		return 10

	elif heur == 'fracdiving' and priority_or_freq == 'freqofs':
		return 10

	elif heur == 'veclendiving' and priority_or_freq == 'freqofs':
		return 4

	else:
		raise Exception('SCIP default not defined')



def get_test_actions(heuristics_to_run, priority_or_freq, episode):
	action = {}

	if priority_or_freq == 'freqofs':
		for heur_to_run in heuristics_to_run:

			if episode == 0:
				action[heur_to_run] = get_scip_default(heur_to_run, priority_or_freq)
			else:
				action[heur_to_run] = episode

	else:

		action_dict_priority = {
			0 : {'coefdiving' : -1001000,
				   'fracdiving' : -1003000,
				   'veclendiving' : -1003100},
			1 : {'coefdiving' : 1,
				   'fracdiving' : 2,
				   'veclendiving' : 3},
			2 : {'coefdiving' : 1,
			       'fracdiving' : 3,
			       'veclendiving' : 2},
			3 : {'coefdiving' : 2,
				   'fracdiving' : 1,
				   'veclendiving' : 3},
			4 : {'coefdiving' : 2,
				   'fracdiving' : 3,
				   'veclendiving' : 1},
			5 : {'coefdiving' : 3,
			       'fracdiving' : 1,
				   'veclendiving' : 2},		
			6 : {'coefdiving' : 3,
			       'fracdiving' : 2,
			       'veclendiving' : 1}
		}

		action = action_dict_priority[episode]

	return action




def main():

	# specify path to data
	#instances = load_generated('facilities')
	instances = load_miplib('easy')



	priority_or_freq = 'priority'
	#priority_or_freq = 'freqofs'

	heuristics_to_run = ['coefdiving', 'fracdiving', 'veclendiving']
	#heuristics_to_run = ['veclendiving']

	# init environment
	env = Enviornment(priority_or_freq=priority_or_freq, 
					 heuristics_to_run=heuristics_to_run)

	num_episodes = 7
	instance_start = 2
	num_instances = 2

	rewards = []
	actions = []

	for instance in instances[instance_start : instance_start + num_instances]:

		for episode in range(0, num_episodes):

			# get actions
			action = get_test_actions(heuristics_to_run, priority_or_freq, episode)

			# reset and step in enviornment 
			env.reset(instance, seed = 0)
			reward = env.step(action)

			rewards.append(reward)
			actions.append(action)

	reward_ts = list(map(lambda x: x['time_to_solve'], rewards))
	reward_nn = list(map(lambda x: x['number_of_nodes'], rewards))

	print('Time to solve:    ', reward_ts)
	print('Number of nodes:  ', reward_nn)
	print('Actions:', actions)


if __name__ == '__main__':
	main()

'''
SCIP Status        : problem is solved [optimal solution found]
Solving Time (sec) : 109.93
Solving Nodes      : 480
Primal Bound       : +1.86289641678731e+04 (270 solutions)
Dual Bound         : +1.86289641678731e+04
Gap                : 0.00 %
Time to solve:     [95.688118, 118.697624, 114.630698, 110.368114, 117.166372, 111.432038, 106.919935, 118.055116, 107.248777, 109.931529]
Number of nodes:   [490, 599, 545, 490, 490, 490, 475, 475, 480, 480]
'''


'''
self.action_dict_priority = {
			0 : {'coefdiving' : -1001000,
				   'fracdiving' : -1003000,
				   'veclendiving' : -1003100},
			1 : {'coefdiving' : 1,
				   'fracdiving' : 2,
				   'veclendiving' : 3},
			2 : {'coefdiving' : 1,
			       'fracdiving' : 3,
			       'veclendiving' : 2},
			3 : {'coefdiving' : 2,
				   'fracdiving' : 1,
				   'veclendiving' : 3},
			4 : {'coefdiving' : 2,
				   'fracdiving' : 3,
				   'veclendiving' : 1},
			5 : {'coefdiving' : 3,
			       'fracdiving' : 1,
				   'veclendiving' : 2},		
			6 : {'coefdiving' : 3,
			       'fracdiving' : 2,
			       'veclendiving' : 1}
		}

	self.scip_default_freqs = {
			'coefdiving' : 10,
			'fracdiving' : 10,
			'veclendiving' : 4
		}

'''

'''

SCIP Status        : problem is solved [optimal solution found]
Solving Time (sec) : 63.00
Solving Nodes      : 269
Primal Bound       : +1.76428171427203e+04 (280 solutions)
Dual Bound         : +1.76428171427203e+04
Gap                : 0.00 %
Time to solve:     [36.0, 39.0, 38.0, 40.0, 40.0, 40.0, 40.0, 63.0, 62.0, 62.0, 64.0, 63.0, 62.0, 63.0]
Number of nodes:   [40, 40, 40, 40, 40, 40, 40, 269, 269, 269, 269, 269, 269, 269]
Actions: [{'coefdiving': -1001000, 'fracdiving': -1003000, 'veclendiving': -1003100}, {'coefdiving': 1, 'fracdiving': 2, 'veclendiving': 3}, {'coefdiving': 1, 'fracdiving': 3, 'veclendiving': 2}, {'coefdiving': 2, 'fracdiving': 1, 'veclendiving': 3}, {'coefdiving': 2, 'fracdiving': 3, 'veclendiving': 1}, {'coefdiving': 3, 'fracdiving': 1, 'veclendiving': 2}, {'coefdiving': 3, 'fracdiving': 2, 'veclendiving': 1}, {'coefdiving': -1001000, 'fracdiving': -1003000, 'veclendiving': -1003100}, {'coefdiving': 1, 'fracdiving': 2, 'veclendiving': 3}, {'coefdiving': 1, 'fracdiving': 3, 'veclendiving': 2}, {'coefdiving': 2, 'fracdiving': 1, 'veclendiving': 3}, {'c
'''