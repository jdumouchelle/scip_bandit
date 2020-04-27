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


def generate_data_offline(
	#instance_set,
	priority_or_freq = 'freq', 
	heuristics_to_run = ['veclendiving'], 
	seeds = [0,1,2,3,4],
	actions = [{'veclendiving' : -1}, {'veclendiving' : 1}, {'veclendiving' : 10}],
	max_rows = 100000, 
	max_cols = 100000,
	time_in_mins = 5):
	'''
		Genreates the data for solving MIP instances 
	'''

	# load instances
	instances = get_miplib_2010_2017_intersection(max_rows, max_cols)

	# init environment
	time_limit = 60 * time_in_mins
	env = Enviornment(priority_or_freq=priority_or_freq, 
					 heuristics_to_run=heuristics_to_run, 
					 time_limit=time_limit)

	# define instance start and end
	instance_start = 0
	instance_end = len(instances)
	total_instances = len(instances)

	# 
	rewards = []
	inst = []
	actions_taken = []
	d = {}
	
	# generate data across instances, actions, seeds
	print('Genereating for instances', instance_start, 'to', instance_end, '(i.e.', total_instances, 'instances)')
	for instance, objective_value in instances[instance_start : instance_end]:
		for action in actions:
			for seed in seeds:


				result_write_path = get_scip_results_path(instance, priority_or_freq, action, seed)
				reward_write_path = get_rewards_path(instance, priority_or_freq, action, seed)

				# skip if data already exists
				if os.path.exists(write_path):
					print('Skipping:', write_path)
					continue
	
				print('Running:', write_path)

				# reset and step in enviornment 
				state = env.reset(instance, seed = seed)
				reward = env.step(action, result_write_path, objective_value)

				# include some information about the instance, and store in a pickle file.
				reward['instance'] = instance 
				reward['objective_value'] = objective_value
				reward['action'] = action
				reward['action'] = action
				reward['state'] = state

				with open(reward_write_path, 'wb') as p:
					pickle.dump(reward, p)

				rewards.append(reward)
				actions_taken.append(action)
				inst.append(instance)

				d[instance] = rewards

	
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



if __name__ == '__main__':

	#parser = argparse.ArgumentParser(description='Determines the subset of instances which data should be generated for.')
	#parser.add_argument('instance_set', metavar='N', type=int,
	#				help='an integer to select set of instances, in the range of 0-8')
	#args = parser.parse_args()
	generate_data_offline()
