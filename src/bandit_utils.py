import os
import sys
import numpy as np
import pandas as pd
import pyscipopt
import pickle
from pyscipopt import Model
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

from enviornment import Enviornment
from reward import Reward

from data_loader import *
from utils import *


def get_instances(actions = [{'veclendiving' : -1}, {'veclendiving' : 1}, {'veclendiving' : 10}], 
                  seeds = [0,1,2,3,4], 
                  priority_or_freq = 'freq', 
                  max_primal_integral=300,
                  min_time_to_solve = 5,
                  max_rows = 100000,
                  max_cols = 100000,
                  reward_path_prefix = '../solved_mip_results/rewards_5_min/'):
    '''
        Gets the set of all instances.  Note that it filters instances which find a 
        primal solution within specified time limit across all seeds and actions.  In 
        addition, it excludes instances which are too easy and solved in less than 
        min_time_to_solve.  
        Params:
            actions - a list of all the actions that the store data is with respect to.
            seeds - a list of all the seeds that the stored data is with respect to.
            priority_or_freq - a string of 'priority' or 'freq' indicating the action space.
            max_primal_integral - the maximal value of the primal integral (used to filter out no solution results).
            min_time_to_solve - the minimum number of seconds to solve the MIP instance.
            max_rows - the maximum number of rows for the MIP instances.
            max_cols - the maximum number of columns for the MIP instances. 
        Returns:
            a dictionary with key value tuple as (instance, state)
    '''
    
    # load instances
    instances = get_miplib_2010_2017_instances(max_rows, max_cols)

    # dict to store instances which find bound within time limit
    instances_to_be_included = {}
    instances_to_be_removed = []
    
    # iterate through instances
    for instance, _ in instances:
        for action in actions:            
            for seed in seeds:
            
                # load results from solved mip instance
                reward_path = get_rewards_path(instance, priority_or_freq, action, seed, reward_path_prefix)                
                with open(reward_path, 'rb') as p:
                    reward = pickle.load(p)
                    
                # skip instances if no primal bound was found
                if reward['primal_integral']  > max_primal_integral - 0.001:
                    continue
                    
                # add instance if instance was solved easily (under min_time_to_solve)
                if reward['time_to_solve'] < min_time_to_solve or instance in instances_to_be_removed:
                    instances_to_be_removed.append(instance)
                    continue
                    
                # add instance, state to dict if not already in
                if instance not in instances_to_be_included:
                    instances_to_be_included[instance] = reward['state']
         
                        
    return instances_to_be_included



def get_solved_mip_result_dict(instances, priority_or_freq, actions, action_values, seeds, reward_path_prefix):
    '''
    '''
    solved_mip_result_dict = {}

    for instance in instances.keys():

        solved_mip_result_dict[instance] = {}

        for action, action_value in zip(actions, action_values):

            solved_mip_result_dict[instance][action_value] = {}
            
            for seed in seeds:
                
                # get primal integral
                reward_path = get_rewards_path(instance, priority_or_freq, action, seed, prefix=reward_path_prefix)
                with open(reward_path, 'rb') as p:
                    reward = pickle.load(p)
                primal_integral = reward['primal_integral']

                # store primal integral in dict
                solved_mip_result_dict[instance][action_value][seed] = - primal_integral

    return solved_mip_result_dict

def get_train_test_split(instances, tr_te_split_ratio=0.8, seed = 0):
    '''
        Gets a train, test split from the data randomly.
        Params:
            instances - a dictionary of the instances (instance:state) pairs.
            tr_te_split_ratio - the train/test ratio of the data.  
            seed - seed for fixing the permutation.
        Returns:
            two dictionaries split randomly as train and test data. 
    '''
    
    # get train/test indicies
    np.random.seed(seed)
    perm = np.random.permutation(len(instances))
    train_idx = perm[:int(tr_te_split_ratio*(len(instances)))]
    test_idx = perm[int(tr_te_split_ratio*(len(instances))):]

    # fill train/test sets
    count = 0
    train_instances = {}
    test_instances = {}
    
    for instance, state in instances.items():
        if count in train_idx:
            train_instances[instance] = state
        else:
            test_instances[instance] = state
        count += 1
        
    return train_instances, test_instances



def get_random_instance(instances):
    '''
        Chooses a random instance from the set of train and test instances.  
        Params:
            instances - a dict of the instances.
        Returns:
            a tuple of the instance name and the state as a numpy array. 
    '''
    
    # select a random instances
    rand_instance = np.random.randint(len(instances))
    
    instance = list(instances.keys())[rand_instance]
    state = instances[instance]
    
    state_as_arr = np.array(list(state.values()))
    
    return instance, state_as_arr


def take_action_on_instance(
    solved_mip_result_dict,
    instance, 
    action, 
    seed=None, 
    priority_or_freq = 'freq', 
    heuristic = 'veclendiving'):
    '''
        Takes action on instance.
        Params:
            instance - a dict of the instances.
            action - the action to take
            seed - a seed for choosing the instance, if not specified then choose one at random.
        Returns:
            the negative primal integral
    '''
    
    # choose seed if not specified
    if seed is None:
        seed = np.random.randint(5)
        
    return solved_mip_result_dict[instance][action][seed]


def get_scaler_normalize_train_states(train_instances):
    '''
        Get a scaler for the input. 
        Params:
            train_instances - a dict of the training instances. 
        Returns:
            returns a sklearn scaler.   
    '''
    train_states = list(train_instances.values())

    train_states = list(map(lambda x: (list(x.values())), train_states))
    train_states = np.array(train_states)
    
    scaler = MinMaxScaler()
    scaler.fit(train_states)
    
    return scaler


def compute_scip_action_reward(instances, solved_mip_result_dict, seeds=[0,1,2,3,4]):
    '''
        Computes the optimal reward for each instance by checking all actions.  
        Params:
            instances - 
            seeds - 
        Returns:
            the optimal reward given the action
    '''

    scip_rewards = []
    
    scip_action = 10
    
    
    for instance in instances.keys():
            
        # compute reward across all instances
        scip_reward = 0
        for seed in seeds:
            reward = take_action_on_instance(solved_mip_result_dict, instance, scip_action, seed)
            scip_reward += reward
        scip_reward = scip_reward / len(seeds)
            
        scip_rewards.append(scip_reward)
        
    scip_avg_reward = np.mean(scip_rewards)
    
    return scip_avg_reward


def compute_optimal_action_reward(instances, actions, solved_mip_result_dict, seeds=[0,1,2,3,4]):
    '''
        Computes the optimal reward for each instance by checking all actions.  
        Params:
            instances - 
            actions - 
            seeds - 
        Returns:
            the optimal reward given the action
    '''

    opt_instance_rewards = []
    
    for instance in instances.keys():
        
        opt_action = -1
        opt_action_reward = -1e7
        
        for action in actions:
            
            # compute reward across all instances
            instance_action_reward = 0
            for seed in seeds:
                reward = take_action_on_instance(solved_mip_result_dict, instance, action, seed)
                instance_action_reward += reward
            instance_action_reward = instance_action_reward / len(seeds)
            
            if instance_action_reward > opt_action_reward:
                opt_action_reward = instance_action_reward
                opt_action = action
                
        opt_instance_rewards.append(opt_action_reward)
        
    opt_val = np.mean(opt_instance_rewards)
    return opt_val





def eval_on_all(bandit, instances, solved_mip_result_dict, seeds = [0,1,2,3,4]):
    '''
        Greedily evaluates the bandit on all specified instances.  The evaluation is run across
        all seeds.
        Params:
            bandit - a trained bandit
            instances - a dict of the instances
        Returns:
            a tuple of the average reward, and the actions
    '''
    
    total_reward = 0
    actions = []
    
    for instance, state in instances.items():
        
        ep_reward = 0
        for seed in seeds:
            state_as_arr = np.array(list(state.values()))

            # take greedy action 
            action = bandit.get_best_action(state_as_arr)
            reward = take_action_on_instance(solved_mip_result_dict, instance, action, seed)

            ep_reward += reward
        avg_ep_reward = ep_reward / len(seeds)
        
        total_reward += avg_ep_reward
        
    avg_reward = total_reward / len(instances)
    
    return avg_reward, actions


def scale_reward(reward, max_primal_integral):
    '''
        scales the rewards between 0 and 1.
        Params:
            reward - the true observered reward.
            max_primal_integral - the maximum value for a primal integral, which is defined through the time limit.  
    '''
    scaled_reward = reward / max_primal_integral + 1

    return scaled_reward

