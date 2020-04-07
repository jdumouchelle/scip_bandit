import os
import sys
import numpy as np
import pyscipopt
from pyscipopt import Model




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
	