import numpy as np 
import math
import os 
import sys
import pickle

from data_loader import *


def get_scip_results_path(instance, priority_or_freq, action, seed, prefix = '../solved_mip_results/results_5_min/'):
	'''
		Gets the path that the scip results should be written to.
		Params:
			instance - file path to the instances 
			priority_or_freq - the specified enviornment, i.e. "priority" or "freq"
			action - the dictionary specifying the action
			seed - the seed
			prefix - the path to where the data is located.  
		Returns:
			the path as a string
	'''
	instance_name = instance.split('/')[-1].split('.')[0]
	write_path = prefix + instance_name + '_' + priority_or_freq + '_seed' + str(seed) + '_'
	for heur, val in action.items():
		write_path += heur + '_' + str(val)
	write_path += '.txt'

	return write_path


def get_rewards_path(instance, priority_or_freq, action, seed, prefix = '../solved_mip_results/rewards_5_min/'):
	'''
		Gets the path to read of write rewards to with respect to a given set of parameters.
		Params:
			instance - file path to the instances 
			priority_or_freq - the specified enviornment, i.e. "priority" or "freq"
			action - the dictionary specifying the action
			seed - the seed
			prefix - the path to where the data is located.  
		Returns:
			the path as a string
	'''
	instance_name = instance.split('/')[-1].split('.')[0]
	write_path = prefix + instance_name + '_' + priority_or_freq + '_seed' + str(seed) + '_'
	for heur, val in action.items():
		write_path += heur + '_' + str(val)
	write_path += '.pickle'

	return write_path



def get_miplib_2010_2017_intersection(max_rows, max_cols):
	'''
		Loads the intersections of MIPLIB easy, benchmark instances from MIPLIB 
		2010 and 2017.  
		Params:
			max_rows - the maximum number of rows to include in an instance
			max_cols - the maximum number of columns to include in an instance.  
		Returns:
			a list containing the union of file paths to instances. 
	'''
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

	return instances