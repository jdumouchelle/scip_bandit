import os
import sys
import numpy as np
import pyscipopt
from pyscipopt import Model




def load_miplib(instance_type = 'easy'):
	'''
		Returns file paths to all MIPLIB instance for a given category.
		Params:
			instance_type - type of instance (easy, hard, or open)
		Returns:
			a list of the path to every file. 
	'''
	file_path = 'data/miplib_2017/' + instance_type + '.txt'

	with open(file_path, 'r') as f:
		files = f.readlines()

	files = list(map(lambda x: x[0:-2], files)) # remove newline

	file_paths = list(map(lambda x: 'data/miplib_2017/collection/' + x, files))

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
	
	fp = 'data/miplib_2017/gz_test.mps'
	model = Model()
	model.readProblem(fp)

	'''
	# test reading of miplib instances
	print('Testing miplib loader...')

	miplib_instances = load_miplib('easy')
	instance = miplib_instances[0]

	print('  Reading instance', instance)
	model = Model()
	model.readProblem(instance)

	print('  Success')


	# test reading of generated instances
	print('Testing generated loader...')
	generated_instances = load_generated('facilities')
	instance = generated_instances[0]

	print('  Reading instance', instance)
	model = Model()
	#model.readProblem(instance)

	print('  Success')
	#print(miplib_instances)
	#print(generated_instances)
	'''

if __name__ == '__main__':
	main()
	