import os
import sys
import numpy as np
import pandas as pd
import pyscipopt
from pyscipopt import Model



class MIPLib2010Loader(object):

	def __init__(self, 
				status = 'easy', 
				sets = ['B', 'P', 'U', 'R', 'T'],
				sets_excluded = ['C', 'I', 'X'], # exclude challenge, infeasible, and large instances
				problem_types = ['BP', 'IP', 'MBP', 'MIP'],
				max_rows = 1e9,
				max_cols = 1e9):
		'''
			Constructor for MIPLIB2010 instances.
			Params:
				status - the diffculty of the problem
				sets - the set of instances to include
				problem_types - the type of MIP instances
				max_rows - the maximum number of cols for an instance
				max_cols - the maximum number of rows for an instance
		'''

		self.year = '2010'
		self.status = status
		self.sets = sets
		self.sets_excluded = sets_excluded
		self.problem_types = problem_types
		self.max_rows = max_rows
		self.max_cols = max_cols

		self.data_info_file = '../data/miplib_' + self.year + '/miplib_info.csv'
		
		self.data_info = pd.read_csv(self.data_info_file)
		self.filter_data()
		
		self.data_path = '../data/miplib_' + self.year + '/collection/'
		return


	def filter_data(self):
		'''
			Filters out instances that do not meet criteria specified in constructor.  
		'''

		# Filter by difficulty status
		if self.status is not None:
			self.data_info = self.data_info[self.data_info['Status'] == self.status]
			
		# Remove all specified problems sets
		for item in self.sets_excluded:
			self.data_info = self.data_info[~ self.data_info['Sets'].str.contains(item)]
			
		# Keep all specified problem sets
		if self.sets is not None:
			idx_to_keep = []
			for item in self.sets:
				idx = self.data_info[self.data_info['Sets'].str.contains(item)].index.tolist()
				idx_to_keep += idx
			
			# remove duplicates and drop indicies
			idx_to_keep = list(set(idx_to_keep))
			self.data_info = self.data_info.loc[idx_to_keep]
			
		# Filter by problem type
		self.data_info = self.data_info[self.data_info['C'].isin(self.problem_types)]
			
		# Filter by max rows
		self.data_info = self.data_info[self.data_info['Rows'] < self.max_rows]
		
		# Filter by max cols
		self.data_info = self.data_info[self.data_info['Cols'] < self.max_cols]
		
		return
	
	def get_instances(self):
		
		files = self.data_info['Name'].tolist()
		file_paths = list(map(lambda x: self.data_path + x + '.mps', files))
		
		opt_solutions = self.data_info['Objective']

		instances = list(map(lambda x,y: (x,float(y)), file_paths, opt_solutions))

		return instances



class MIPLib2017Loader(object):

	def __init__(self, 
				status = 'easy', 
				sets = ['benchmark_suitable'],
				sets_excluded = ['infeasible'],
				max_rows = 1e9,
				max_cols = 1e9):

		'''
			Constructor for MIPLIB2010 instances.
			Params:
				status - the diffculty of the problem
				sets - the set of instances to include
				problem_types - the type of MIP instances
				max_rows - the maximum number of cols for an instance
				max_cols - the maximum number of rows for an instance
		'''
		self.year = '2017'
		self.status = status
		self.sets = sets
		self.sets_excluded = sets_excluded
		self.max_rows = max_rows
		self.max_cols = max_cols

		self.data_info_file = '../data/miplib_' + self.year + '/miplib_info.csv'
		
		self.data_info = pd.read_csv(self.data_info_file)
		self.filter_data()
		
		self.data_path = '../data/miplib_' + self.year + '/collection/'

		return


	def filter_data(self):
		
		'''
			Filters out instances that do not meet criteria specified in constructor.  
		'''
		
		# Filter by difficulty status
		if self.status is not None:
			self.data_info = self.data_info[self.data_info['Status'] == self.status]
		
		# Remove all specified problems sets
		if self.sets_excluded is not None:
			idx_to_keep = []
			for item in self.sets_excluded:
				idx = self.data_info[self.data_info[item] == 0].index.tolist()
				idx_to_keep += idx

			# remove duplicates and drop indicies
			idx_to_keep = list(set(idx_to_keep))
			self.data_info = self.data_info.loc[idx_to_keep]
		
		# Keep all specified problem sets
		if self.sets is not None:
			idx_to_keep = []
			for item in self.sets:
				idx = self.data_info[self.data_info[item] == 1].index.tolist()
				idx_to_keep += idx
			
			# remove duplicates and drop indicies
			idx_to_keep = list(set(idx_to_keep))
			self.data_info = self.data_info.loc[idx_to_keep]
		
		# Filter by max rows
		self.data_info = self.data_info[self.data_info['Constraints'] < self.max_rows]
		
		# Filter by max cols
		self.data_info = self.data_info[self.data_info['Variables'] < self.max_cols]
		
		return
	
	def get_instances(self):
		'''
			Gets the instances that fit critera specified in constructor.  
			Returns:
				a list of tuples of the (path_to_instance, objective_value)
		'''
		files = self.data_info['Instance'].tolist()
		file_paths = list(map(lambda x: self.data_path + x + '.mps', files))
		
		opt_solutions = self.data_info['Objective']

		instances = list(map(lambda x,y: (x,float(y)), file_paths, opt_solutions))

		return instances


def load_test():
	'''
		Loads a sample instance
	'''
	return  [('../data/tests/test.mps', None)]



def main():
	# test reading of miplib instances
	print('Testing miplib loader...')

	miplib_instances = load_test()

	instance, _ = miplib_instances[0]

	print('  Reading instance', instance)
	model = Model()
	model.readProblem(instance)
	print('  Success')
	
	return
	
if __name__ == '__main__':
	main()
	