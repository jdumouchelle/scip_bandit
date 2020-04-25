import numpy as np 
import math
import os 
import sys
import pickle
import argparse

from pyscipopt import Model

from enviornment import Enviornment
from data_loader import *
from reward import Reward
from utils import *

import win32file
win32file._setmaxstdio(4096)


def main(instance_set):

	# load instances
	max_rows = 100000
	max_cols = 100000

	data_loader_2010 = MIPLib2010Loader(max_rows = max_rows, max_cols = max_cols)
	instances_2010 = data_loader_2010.get_instances()

	data_loader_2017 = MIPLib2017Loader(max_rows = max_rows, max_cols = max_cols)
	instances_2017 = data_loader_2017.get_instances()

	instance_names_2010 = list(map(lambda x: x[0], instances_2010))
	instance_names_2017 = list(map(lambda x: x[0], instances_2017))

	unique_2010_instances = list(map(lambda x: x.replace('miplib_2010', 'miplib_2017'), instance_names_2010))
	unique_2010_instances = list(filter(lambda x: x not in instance_names_2017, unique_2010_instances) )
	unique_2010_instances = list(map(lambda x: x.replace('miplib_2017', 'miplib_2010'), unique_2010_instances))

	instances_2010_filtered = list(filter(lambda x: x[0] in unique_2010_instances, instances_2010))

	instances = instances_2017 + instances_2010_filtered

	# set enviornment conditions
	mins = 5
	time_limit = 60 * mins
	priority_or_freq = 'freq'
	heuristics_to_run = ['veclendiving']

	# init environment
	env = Enviornment(priority_or_freq=priority_or_freq, 
					 heuristics_to_run=heuristics_to_run, 
					 time_limit=time_limit)

	# init number of epsiodes, instances, rewards, actions
	num_procs = 8
	num_in_set = int(len(instances) / num_procs) + 1

	instance_start = instance_set * num_in_set
	instance_end = instance_set * num_in_set + num_in_set
	if instance_end > len(instances):
		instance_end = len(instances)
	
	total_instances = instance_end - instance_start

	# define actions
	actions = [{'veclendiving' : -1},
		{'veclendiving' : 1},
		{'veclendiving' : 10}]

	seeds = [0,1,2,3,4] # test over 5 different seeds

	rewards = []
	inst = []
	actions_taken = []
	d = {}
	
	
	print('Genereating for instances', instance_start, 'to', instance_end, '(i.e.', total_instances, 'instances)')
	
	#for instance, _ in instances[instance_start : instance_end]:
	#	print(instance)
	#return
	num_exists = 0
	num_total = 0
	for instance, objective_value in instances[instance_start : instance_end]:
		#print(instance)
		for action in actions:
			#print('  ', action)
			for seed in seeds:

				#instance = 'data/miplib_2010/collection/satellites1-25.mps'
				#action = {'veclendiving' : -1}
				#seed = 0

				write_path = get_scip_write_path(instance, priority_or_freq, action, seed)
				if os.path.exists(write_path):
					print('Skipping:', write_path)
					continue
				else:
					print('Running:', write_path)

				# reset and step in enviornment 
				state = env.reset(instance, seed = seed)

				reward = env.step(action, write_path, objective_value)

				reward['instance'] = instance 
				reward['objective_value'] = objective_value
				reward['action'] = action
				reward['action'] = action
				reward['state'] = state

				reward_write_path = get_rewards_write_path(instance, priority_or_freq, action, seed)
				with open(reward_write_path, 'wb') as p:
					pickle.dump(reward, p)

				rewards.append(reward)
				actions_taken.append(action)
				inst.append(instance)

				d[instance] = rewards
				#return

	print(num_exists, num_total)
	
	time_to_solve = list(map(lambda x: x['time_to_solve'], rewards))
	number_of_nodes = list(map(lambda x: x['number_of_nodes'], rewards))
	primal_dual_gap = list(map(lambda x: x['primal_dual_gap'], rewards))
	lp_iterations = list(map(lambda x: x['LP_iterations'], rewards))
	primal_bounds = list(map(lambda x: x['primal_bounds'], rewards))
	primal_gaps = list(map(lambda x: x['primal_gaps'], rewards))
	primal_integral = list(map(lambda x: x['primal_integral'], rewards))

	print('Time to solve:    ', time_to_solve)
	print('Number of nodes:  ', number_of_nodes)
	print('Primal dual gap:  ', primal_dual_gap)
	print('Primal Integral:  ', primal_integral)
	print('Actions:          ', actions_taken)
	print('Instance:         ', inst)

	print(objective_value)

	with open('results' + str(instance_set) + '.pickle', 'wb') as p:
		pickle.dump(rewards, p)


if __name__ == '__main__':

	parser = argparse.ArgumentParser(description='Parse set of instances')
	parser.add_argument('set', metavar='N', type=int,
					help='an integer to select set of instances')
	args = parser.parse_args()
	main(args.set)

	#for i in range(0,8):
	#		main(i)