import os
import sys
import numpy as np
import pandas as pd
import pyscipopt
import pickle
from pyscipopt import Model
from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt



from data_loader import *
from utils import *
from bandit_utils import *


from enviornment import Enviornment
from reward import Reward



class LinUCB(object):
    
    
    def __init__(self, actions, state_dim, alpha):
        '''
            actions - a list of the actions as strings.
            state_dim - the state dimension of the problem.
            alpha_dim - coefficient for UCB.
        '''
        
        self.actions = actions
        self.state_dim = state_dim
        self.alpha = alpha 
        
        self.theta_a = {}
        self.A_a = {}
        self.b_a = {}
        
        # initialize linear weights
        for action in self.actions:    
            
            self.theta_a[action] = np.zeros(self.state_dim)
            self.A_a[action] = np.eye(self.state_dim)
            self.b_a[action] = np.zeros(self.state_dim)
            
        self.t = 0
        
        
    def get_action(self, state): 
        '''
            Greedily gets an action based on store parameters and UCB.
            Params:
                state - a numpy array of the state
            Returns:
                the action assosiated with the highest expected return.
        '''
        
        action_scores = np.zeros(len(self.actions))
        
        for i  in range(len(self.actions)):
            
            action = self.actions[i]
            
            A_a_inv = np.linalg.inv(self.A_a[action])
            b_a = self.b_a[action]
            theta_hat_a = np.dot(A_a_inv, b_a)
            p_a = np.dot(theta_hat_a, state) + self.alpha*np.sqrt(np.dot(state.T, np.dot(A_a_inv, state)))
            
            action_scores[i] = p_a
                
        # break ties arbitrarily
        best_action_index = np.random.choice(np.flatnonzero(action_scores == action_scores.max()))
        best_action = self.actions[best_action_index]
        
        return best_action
    
    
        
    def get_best_action(self, state): 
        '''
            Greedily gets an action based on store parameters.  Does not use UCB.
            Params:
                state - a numpy array of the state
            Returns:
                the action assosiated with the highest expected return.
        '''
        
        action_scores = np.zeros(len(self.actions))
        
        for i  in range(len(self.actions)):
            
            action = self.actions[i]
            
            A_a_inv = np.linalg.inv(self.A_a[action])
            b_a = self.b_a[action]
            theta_hat_a = np.dot(A_a_inv, b_a)
            p_a = np.dot(theta_hat_a, state) 
            
            action_scores[i] = p_a
            
        # break ties arbitrarily
        best_action_index = np.random.choice(np.flatnonzero(action_scores == action_scores.max()))
        best_action = self.actions[best_action_index]
        
                
        return best_action
    
    
    def update_params(self, reward, state, action):
        '''
            Updates parameters given a reward, state, and action
        '''
        
        self.A_a[action] = self.A_a[action] + np.outer(state, state)
        self.b_a[action] = self.b_a[action] + reward*state
        
        return