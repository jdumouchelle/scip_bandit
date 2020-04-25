import numpy as np 
import math
import os 
import sys
import pickle


def get_scip_default(heur, priority_or_freq):
	'''
		Returns default scip parameters for actions that we are considering in the 
		MDP.  
	'''

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
	'''
		Gets test 
	'''

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



def get_scip_write_path(instance, priority_or_freq, action, seed):
	'''
	'''
	instance_name = instance.split('/')[-1].split('.')[0]
	write_path = 'results/' + instance_name + '_' + priority_or_freq + '_seed' + str(seed) + '_'
	for heur, val in action.items():
		write_path += heur + '_' + str(val)
	write_path += '.txt'

	return write_path


def get_rewards_write_path(instance, priority_or_freq, action, seed):
	'''
	'''
	instance_name = instance.split('/')[-1].split('.')[0]
	write_path = 'rewards/' + instance_name + '_' + priority_or_freq + '_seed' + str(seed) + '_'
	for heur, val in action.items():
		write_path += heur + '_' + str(val)
	write_path += '.pickle'

	return write_path