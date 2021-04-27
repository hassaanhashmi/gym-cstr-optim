#!/usr/bin/env python3
"""
Classical Additive White Gaussian Noise (AWGN) channel
"""
import math
import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np

class AWGN(gym.Env):
    def __init__(self,
                 num_users=5,
                 priority_weights=None,
                 pow_max=20,
                 channel_mu=2,
                 noise_var=1
                 ):
        super(AWGN, self).__init__()
        self.num_users = num_users
        self.pow_max = pow_max
        self.channel_mu = channel_mu
        self.noise_var = noise_var
        self.vec_f_out = np.zeros(shape=(self.num_users,1))
        if np.any(priority_weights == None):
            self. priority_weights = np.ones(shape=(num_users,1))/num_users
        else:
            assert priority_weights.shape[0] == self.num_users, "no. of priority weights != no. of users"
            assert np.sum(priority_weights) == 1, "sum of priority weights != 1"
            self. priority_weights = priority_weights
        
    def sample_fading_channels(self):
        vec_H = np.random.exponential(self.channel_mu, size=(self.num_users,1))
        return vec_H

    def g_o(self, vec_metric_x):
        return np.dot(self.priority_weights.T, vec_metric_x)[0]
    
    def g_(self, vec_actions):
        return self.pow_max - np.sum(vec_actions)

    def vec_f(self, vec_actions, vec_H):
        for i in range(self.num_users):
            self.vec_f_out[i] = np.log(1+ (vec_H[i]*vec_actions[i])/(self.noise_var + np.dot(np.delete(vec_H, i, axis=0).T, np.delete(vec_actions, i, axis=0))))
        return self.vec_f_out

    #gym functions
    def step(self, vec_actions, vec_metrics_x, vec_H):
        vec_metrics_x = np.maximum(0,vec_metrics_x) #projection
        vec_actions = np.minimum(self.pow_max, vec_actions) #projection
        vec_actions = np.maximum(0, vec_actions)
        go_x = self.g_o(vec_metrics_x)
        f_i = self.f_i(vec_actions)
        f_h = self.vec_f(vec_actions, vec_H)
        vec_H = self.sample_fading_channels()
        return go_x, f_i, f_h, vec_H
        
    def reset(self):
        #No states to reset
        return self.sample_fading_channels()

    def render(self):
        #TODO
        pass

    def close(self):
        #TODO
        pass