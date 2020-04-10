import os
import sys
import numpy as np
import pandas as pd
import pyscipopt
from pyscipopt import Model



class MIPLib2010Loader(object):

    def __init__(self, 
                year = 2010,
                status = 'easy', 
                sets = ['B'],
                sets_excluded = ['C', 'I', 'P', 'U', 'R', 'T', 'X'],
                problem_types = ['BP', 'IP', 'MBP', 'MIP'],
                max_rows = 1e9,
                max_cols = 1e9):

        self.year = str(year)
        self.status = status
        self.sets = sets
        self.sets_excluded = sets_excluded
        self.problem_types = problem_types
        self.max_rows = max_rows
        self.max_cols = max_cols

        self.data_info_file = 'data/miplib_' + self.year + '/miplib_info.csv'
        
        self.data_info = pd.read_csv(self.data_info_file)
        self.filter_data()
        
        self.data_path = 'data/miplib_' + self.year + '/collection/'

        
        
        return


    def filter_data(self):
        
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
        
        return file_paths







def load_miplib(instance_type = 'easy', year = 2010):
	'''
		Returns file paths to all MIPLIB instance for a given category.
		Params:
			instance_type - type of instance (easy, hard, or open)
			year - miplib year
		Returns:
			a list of the path to every file. 
	'''

	year = str(year)
	lib_path = 'data/miplib_' + year + '/' 
	file_path = lib_path + instance_type + '.txt'

	with open(file_path, 'r') as f:
		files = f.readlines()

	files = list(map(lambda x: x[:-1], files)) # remove newline

	file_paths = list(map(lambda x: lib_path + 'collection/' + x, files))

	return file_paths




def load_generated(problem_type = 'facilities'):
	'''
		Returns file paths for generated instance.
		Params:
			problem_type - a string indicating the type of problem to focus on.
						   (facilities, indset, setcov, cauction)
	'''

	file_path = 'data/instances/'

	if problem_type == 'facilities':
		file_path += 'facilities/train_100_100_5/'

	else:
		raise Exception('Problem not yet defined, please generate instances.')

	files = os.listdir(file_path)
	file_paths = list(map(lambda x: file_path + x, files))

	return file_paths




def load_test():

	return  ['data/test.mps']



def main():
	
	
	# test reading of miplib instances
	print('Testing miplib loader...')

	miplib_instances = load_miplib('easy')

	for instance in miplib_instances[0:10]:

		print('  Reading instance', instance)
		model = Model()
		model.readProblem(instance)

	print('  Success')


	# test reading of generated instances
	'''
	print('Testing generated loader...')
	generated_instances = load_generated('facilities')
	instance = generated_instances[0]

	print('  Reading instance', instance)
	model = Model()
	model.readProblem(instance)

	print('  Success')
	'''

	return
	
if __name__ == '__main__':
	main()
	