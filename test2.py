import numpy as np 
import math
import os 
import sys

from pyscipopt import Model



class Enviornment(object):

	def __init__(self, reward_type='time_to_solve'):
		
		self.heuristics = ['actconsdiving' ,'bound' ,'clique' ,'coefdiving' ,'completesol' ,'conflictdiving' ,'crossover' ,'dins' ,'distributiondiving' ,'dualval' ,'farkasdiving' ,'feaspump' ,'fixandinfer' ,'fracdiving' ,'gins' ,'guideddiving' ,'zeroobj' ,'indicator' ,'intdiving' ,'intshifting' ,'linesearchdiving' ,'localbranching' ,'locks' ,'lpface' ,'alns' ,'nlpdiving' ,'mutation' ,'multistart' ,'mpec' ,'objpscostdiving' ,'octane' ,'ofins' ,'oneopt' ,'proximity' ,'pscostdiving' ,'randrounding' ,'rens' ,'reoptsols' ,'repair' ,'rins' ,'rootsoldiving' ,'rounding' ,'shiftandpropagate' ,'shifting' ,'simplerounding' ,'subnlp' ,'trivial' ,'trivialnegation' ,'trysol' ,'twoopt' ,'undercover' ,'vbounds' ,'veclendiving' ,'zirounding' ]
		
		self.heuristics_to_run = ['coefdiving', 'fracdiving', 'veclendiving']
		self.heuristics_to_disable = list(filter(lambda x: x not in self.heuristics_to_run, self.heuristics))

		self.action_dict = {
			0 : {'coefdiving' : 1,
				   'fracdiving' : 2,
				   'veclendiving' : 3},
			1 : {'coefdiving' : 1,
			       'fracdiving' : 3,
			       'veclendiving' : 2},
			2 : {'coefdiving' : 2,
				   'fracdiving' : 1,
				   'veclendiving' : 3},
			3 : {'coefdiving' : 2,
				   'fracdiving' : 3,
				   'veclendiving' : 1},
			4 : {'coefdiving' : 3,
			       'fracdiving' : 1,
				   'veclendiving' : 2},		
			5 : {'coefdiving' : 3,
			       'fracdiving' : 2,
			       'veclendiving' : 1}
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
			print(heur_to_disable)
			self.model.setIntParam('heuristics/' + heur_to_disable + '/freqofs', 0)
			self.model.setIntParam('heuristics/' + heur_to_disable + '/priority', -100000)

		return


	def set_priority(self, action):
		'''
			Sets priorities for heurisitics coefdiving, fracdiving, veclendiving 
			according to an action
		'''
		priorities = self.action_dict[action]

		for heur_to_run in self.heuristics_to_run:
			self.model.setIntParam('heuristics/' + heur_to_run + '/freqofs', priorities[heur_to_run])
			print('heuristics/' + heur_to_run + '/freqofs', priorities[heur_to_run])

		return

	def get_state(self):
		'''
			Gets the state of the model.  
		'''

		return 0


	def get_reward(self):
		'''
		'''
		if self.reward_type == 'number_of_nodes':
			reward = - self.model.getNNodes()

		elif self.reward_type == 'time_to_solve':
			reward = - self.model.getSolvingTime()

		return reward

	def reset(self, path, seed=0):
		'''
			Loads a random instance and returns state given by the instance.  
		'''
		self.model = Model() 

		self.model.readProblem(path)
		#x = self.model.addVar("x")
		#y = self.model.addVar("y", vtype="INTEGER")
		#self.model.setObjective(x + y)
		#self.model.addCons(2*x - y*y >= 0)

		self.state = self.get_state()

		return self.state 


	def step(self, action):

		self.disable_heuristics()
		self.set_priority(action)

		self.model.optimize()

		# get reward
		reward = self.get_reward()

		return reward



def main():

	# specify path to data
	path = '/home/justin/Desktop/COMP767/project/data/instances/setcover/train_500r_1000c_0.05d/'
	instances = os.listdir(path)


	f_path = path + instances[0]


	# init environment
	reward_type = 'time_to_solve'
	#reward_type = 'number_of_nodes'
	env = Enviornment(reward_type)
	


	rewards = []

	for episode in range(0,6):

		env.reset(f_path, seed = 0)

		reward = env.step(episode)

		rewards.append(reward)


	print(rewards)


if __name__ == '__main__':
	main()