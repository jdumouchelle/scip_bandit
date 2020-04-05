import numpy as np 
import math
import os 
import sys

from pyscipopt import Model



class Enviornment(object):

	def __init__(self, reward_type='time_to_solve'):
		
		self.heuristics = ['actconsdiving' ,'bound' ,'clique' ,'coefdiving' ,'completesol' ,'conflictdiving' ,'crossover' ,'dins' ,'distributiondiving' ,'dualval' ,'farkasdiving' ,'feaspump' ,'fixandinfer' ,'fracdiving' ,'gins' ,'guideddiving' ,'zeroobj' ,'indicator' ,'intdiving' ,'intshifting' ,'linesearchdiving' ,'localbranching' ,'locks' ,'lpface' ,'alns' ,'nlpdiving' ,'mutation' ,'multistart' ,'mpec' ,'objpscostdiving' ,'octane' ,'ofins' ,'oneopt' ,'proximity' ,'pscostdiving' ,'randrounding' ,'rens' ,'reoptsols' ,'repair' ,'rins' ,'rootsoldiving' ,'rounding' ,'shiftandpropagate' ,'shifting' ,'simplerounding' ,'subnlp' ,'trivial' ,'trivialnegation' ,'trysol' ,'twoopt' ,'undercover' ,'vbounds' ,'veclendiving' ,'zirounding' ]
		
		self.heuristics_to_run = ['coefdiving', 'fracdiving', 'veclendiving']
		self.heuristics_to_run = ['veclendiving']
		self.heuristics_to_disable = list(filter(lambda x: x not in self.heuristics_to_run, self.heuristics))

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

		self.state = None
		self.model = None
		self.reward_type = reward_type

		return

	def seed(self, seed):
		self.seed = seed


	def disable_heuristics(self):
		'''
			Disables all heurisitics except coefdiving, fracdiving, veclendiving
		'''
		for heur_to_disable in self.heuristics_to_disable:
			#print(heur_to_disable)
			self.model.setIntParam('heuristics/' + heur_to_disable + '/freqofs', 65534)
			self.model.setIntParam('heuristics/' + heur_to_disable + '/priority', -100000)

		return


	def set_priority(self, action):
		'''
			Sets priorities for heurisitics coefdiving, fracdiving, veclendiving 
			according to an action
		'''
		priorities = self.action_dict_priority[action]

		for heur_to_run in self.heuristics_to_run:
			self.model.setIntParam('heuristics/' + heur_to_run + '/priority', priorities[heur_to_run]) # set priority
			print('heuristics/' + heur_to_run + '/priority', priorities[heur_to_run])

		return


	def set_freqofs(self, action):
		'''
			Sets frequency of coefdiving, fracdiving, veclendiving.
		'''

		

		for i in range(len(self.heuristics_to_run)):

			heur_to_run = self.heuristics_to_run[i]
			action_value = action[heur_to_run]
			
			# set to scip default
			if action_value == -1:
				action_value = self.scip_default_freqs[heur_to_run]

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
		#self.set_priority(action)
		self.set_freqofs(action)
		self.model.optimize()

		# get reward
		reward = self.get_reward()

		return reward



def main():

	# specify path to data
	path = '/home/justin/Desktop/COMP767/project/data/instances/facilities/train_100_100_5/'
	instances = os.listdir(path)

	# init environment
	env = Enviornment()

	num_episodes = 10
	num_instances = 1

	for instance in instances[0:num_instances]:

		f_path = path + instance

		rewards = []

		for episode in range(0, num_episodes):

			if episode == 0:
				action = {
					#'coefdiving' : -1,
					#'fracdiving' : -1,
					'veclendiving' : -1,
				}

			else:
				action = {
					#'coefdiving' : episode,
					#'fracdiving' : episode,
					'veclendiving' : episode,
				}

			env.reset(f_path, seed = 0)

			reward = env.step(action)

			rewards.append(reward)

	reward_ts = list(map(lambda x: x['time_to_solve'], rewards))
	reward_nn = list(map(lambda x: x['number_of_nodes'], rewards))

	print('Time to solve:    ', reward_ts)
	print('Number of nodes:  ', reward_nn)


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