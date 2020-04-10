import numpy as np 
import math
import os 
import sys
import pickle

from data_loader import load_miplib, load_generated, MIPLib2010Loader, load_test
from pyscipopt import Model



class Enviornment(object):

	def __init__(self, priority_or_freq, heuristics_to_run, time_limit=600):
		'''
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

	def seed(self, seed):
		self.seed = seed

		return


	def disable_heuristics(self):
		'''
			Disables all heurisitics except coefdiving, fracdiving, veclendiving
		'''
		for heur_to_disable in self.heuristics_to_disable:
			self.model.setIntParam('heuristics/' + heur_to_disable + '/freq', -1)
			self.model.setIntParam('heuristics/' + heur_to_disable + '/priority', -100000)

		return


	def set_time_limit(self):
		'''
		'''

		self.model.setRealParam('limits/time', self.time_limit)

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



	def set_freq(self, action):
		'''
			Sets frequency of coefdiving, fracdiving, veclendiving.
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
			Gets the state of the model.  
		'''

		return 0


	def get_reward(self):
		'''
		'''
		
		reward_ts = self.model.getTotalTime()
		reward_nn = self.model.getNNodes()
		gap = self.model.getGap()

		reward = {'time_to_solve' : reward_ts,
				  'number_of_nodes' : reward_nn,
				  'gap' : gap}

		return reward

	def reset(self, path, seed=0):
		'''
			Loads a random instance and returns state given by the instance.  
		'''
		self.model = Model() 
		self.model.readProblem(path)
		self.state = self.get_state()

		return self.state 


	def step(self, action, write_path):

		
		self.disable_heuristics()

		# set priorities
		if self.priority_or_freq == 'priority':
			self.set_priority(action)
		else:
			self.set_freq(action)
		
		self.set_time_limit()

		self.model.optimize()


		self.model.writeStatistics(write_path)

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

	elif heur == 'coefdiving' and priority_or_freq == 'freq':
		return -1

	elif heur == 'fracdiving' and priority_or_freq == 'freq':
		return 10

	elif heur == 'veclendiving' and priority_or_freq == 'freq':
		return 10

	else:
		raise Exception('SCIP default not defined')



def get_test_actions(heuristics_to_run, priority_or_freq, episode):

	

	action = {}

	if priority_or_freq == 'freq':
		for heur_to_run in heuristics_to_run:

			action_episode_dict = {
				0 : get_scip_default(heur_to_run, priority_or_freq), # scip default
				1 : 1,
				2 : 2,
				3 : 5,
				4 : 20,
				5 : 30, 
				6 : -1, # don't run
			}

			action[heur_to_run] = action_episode_dict[episode]
			#if episode == 0:
			#	action = get_scip_default(heur_to_run, priority_or_freq)
			#else:
			#	action[heur_to_run] = episode

	else:

		action_dict_priority = {
			0 : {'coefdiving' : -1,
				   'fracdiving' : -1,
				   'veclendiving' : -1},
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



def get_write_path(instance, priority_or_freq, action):

	instance_name = instance.split('/')[-1].split('.')[0]
	write_path = 'results/' + instance_name + '_' + priority_or_freq + '_' 
	for heur, val in action.items():
		write_path += heur + '_' + str(val)
	write_path += '.txt'

	return write_path



def main():

	# specify path to data

	#instances = load_test()
	#instances = load_generated('facilities')
	#instances = load_miplib('easy')

	max_rows = 1000
	max_cols = 1000
	data_loader = MIPLib2010Loader(max_rows=max_rows, max_cols=max_cols)
	instances = data_loader.get_instances()

	time_limit = 600

	#priority_or_freq = 'priority'
	priority_or_freq = 'freq'

	#heuristics_to_run = ['fracdiving']#, 'veclendiving']
	heuristics_to_run = ['veclendiving']
	#heuristics_to_run = ['coefdiving']
	
	# init environment
	env = Enviornment(priority_or_freq=priority_or_freq, 
					 heuristics_to_run=heuristics_to_run, 
					 time_limit=time_limit)

	num_episodes = 7
	instance_start = 2
	#num_instances = 2
	num_instances = len(instances) - instance_start

	rewards = []
	actions = []
	inst = []
	d = {}

	print('Number of instances:', num_instances)
	#x = '1 ' + 1

	for instance in instances[instance_start : instance_start + num_instances]:

		for episode in range(0, num_episodes):

			# get actions
			action = get_test_actions(heuristics_to_run, priority_or_freq, episode)
			write_path = get_write_path(instance, priority_or_freq, action)

			# reset and step in enviornment 
			env.reset(instance, seed = 0)
			reward = env.step(action, write_path)

			rewards.append(reward)
			actions.append(action)
			inst.append(instance)

			d[instance] = rewards

	reward_ts = list(map(lambda x: x['time_to_solve'], rewards))
	reward_nn = list(map(lambda x: x['number_of_nodes'], rewards))
	gap = list(map(lambda x: x['gap'], rewards))

	print('Time to solve:    ', reward_ts)
	print('Number of nodes:  ', reward_nn)
	print('Optimality gap:   ', gap)
	print('Actions:          ', actions)
	print('Instance:         ', inst)


	#with open('scip_default_results.pickle', 'wb') as p:
	#	pickle.dump(d, p)


if __name__ == '__main__':
	main()


'''
heuristics/veclendiving/freq 10
heuristics/veclendiving/freq 1
heuristics/veclendiving/freq 10
heuristics/veclendiving/freq 1
heuristics/veclendiving/freq 10
heuristics/veclendiving/freq 1
heuristics/veclendiving/freq 10
heuristics/veclendiving/freq 1
Time to solve:     [521.0, 515.0, 93.0, 80.0, 600.0, 600.0, 600.0, 600.0]
Number of nodes:   [58273, 56422, 13156, 11837, 137753, 159217, 77343, 57470]
Optimality gap:    [0.0, 0.0, 0.0, 0.0, 0.1111111111111109, 0.11111111111111067, 0.03837721379639923, 0.0388048473382073]
Actions: [{'veclendiving': 10}, {'veclendiving': 1}, {'veclendiving': 10}, {'veclendiving': 1}, {'veclendiving': 10}, {'veclendiving': 1}, {'veclendiving': 10}, {'veclendiving': 1}]


Time to solve:     [600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 112.0, 106.0, 100.0, 108.0, 97.0, 119.0, 123.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 600.0, 354.0, 418.0, 408.0, 445.0, 573.0, 376.0, 310.0]
Number of nodes:   [96600, 70790, 74298, 61735, 86916, 89591, 80109, 50111, 50550, 50344, 51274, 64660, 101855, 107061, 23091, 21173, 21173, 22956, 20335, 25723, 27394, 1, 1, 1, 1, 1, 1, 1, 24442, 29361, 28130, 13570, 7313, 7242, 9615, 79690, 87922, 87922, 89008, 132470, 95674, 114920]
Optimality gap:    [0.1111111111111109, 0.1111111111111109, 0.1111111111111109, 0.1111111111111109, 0.1111111111111109, 0.1111111111111109, 0.1111111111111109, 0.03974324400291678, 0.039185716539196114, 0.03920067879408018, 0.03960970517342826, 0.039169591404200614, 0.03849015506420277, 0.03737173979994341, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5609386991972598, 0.5509585672994367, 0.5542958711013977, 0.6291386378922272, 0.6722026503085297, 0.71452485578396, 0.743793741361312, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
Actions:           [{'veclendiving': 10}, {'veclendiving': 1}, {'veclendiving': 2}, {'veclendiving': 5}, {'veclendiving': 20}, {'veclendiving': 30}, {'veclendiving': -1}, {'veclendiving': 10}, {'veclendiving': 1}, {'veclendiving': 2}, {'veclendiving': 5}, {'veclendiving': 20}, {'veclendiving': 30}, {'veclendiving': -1}, {'veclendiving': 10}, {'veclendiving': 1}, {'veclendiving': 2}, {'veclendiving': 5}, {'veclendiving': 20}, {'veclendiving': 30}, {'veclendiving': -1}, {'veclendiving': 10}, {'veclendiving': 1}, {'veclendiving': 2}, {'veclendiving': 5}, {'veclendiving': 20}, {'veclendiving': 30}, {'veclendiving': -1}, {'veclendiving': 10}, {'veclendiving': 1}, {'veclendiving': 2}, {'veclendiving': 5}, {'veclendiving': 20}, {'veclendiving': 30}, {'veclendiving': -1}, {'veclendiving': 10}, {'veclendiving': 1}, {'veclendiving': 2}, {'veclendiving': 5}, {'veclendiving': 20}, {'veclendiving': 30}, {'veclendiving': -1}]
Instance:          ['data/miplib_2010/collection/cov1075.mps', 'data/miplib_2010/collection/cov1075.mps', 'data/miplib_2010/collection/cov1075.mps', 'data/miplib_2010/collection/cov1075.mps', 'data/miplib_2010/collection/cov1075.mps', 'data/miplib_2010/collection/cov1075.mps', 'data/miplib_2010/collection/cov1075.mps', 'data/miplib_2010/collection/danoint.mps', 'data/miplib_2010/collection/danoint.mps', 'data/miplib_2010/collection/danoint.mps', 'data/miplib_2010/collection/danoint.mps', 'data/miplib_2010/collection/danoint.mps', 'data/miplib_2010/collection/danoint.mps', 'data/miplib_2010/collection/danoint.mps', 'data/miplib_2010/collection/dfn-gwin-UUM.mps', 'data/miplib_2010/collection/dfn-gwin-UUM.mps', 'data/miplib_2010/collection/dfn-gwin-UUM.mps', 'data/miplib_2010/collection/dfn-gwin-UUM.mps', 'data/miplib_2010/collection/dfn-gwin-UUM.mps', 'data/miplib_2010/collection/dfn-gwin-UUM.mps', 'data/miplib_2010/collection/dfn-gwin-UUM.mps', 'data/miplib_2010/collection/enlight13.mps', 'data/miplib_2010/collection/enlight13.mps', 'data/miplib_2010/collection/enlight13.mps', 'data/miplib_2010/collection/enlight13.mps', 'data/miplib_2010/collection/enlight13.mps', 'data/miplib_2010/collection/enlight13.mps', 'data/miplib_2010/collection/enlight13.mps', 'data/miplib_2010/collection/newdano.mps', 'data/miplib_2010/collection/newdano.mps', 'data/miplib_2010/collection/newdano.mps', 'data/miplib_2010/collection/newdano.mps', 'data/miplib_2010/collection/newdano.mps', 'data/miplib_2010/collection/newdano.mps', 'data/miplib_2010/collection/newdano.mps', 'data/miplib_2010/collection/mik.250-1-100.1.mps', 'data/miplib_2010/collection/mik.250-1-100.1.mps', 'data/miplib_2010/collection/mik.250-1-100.1.mps', 'data/miplib_2010/collection/mik.250-1-100.1.mps', 'data/miplib_2010/collection/mik.250-1-100.1.mps', 'data/miplib_2010/collection/mik.250-1-100.1.mps', 'data/miplib_2010/collection/mik.250-1-100.1.mps']


'''
