import numpy as np


class Barrier(object):
    '''
    This class prevents the agent from
    going outside the exploration space
    '''
    def __init__(self, explr_space, pow=2, weight=100):
        self.explr_space = explr_space
        self.dl  = explr_space.high - explr_space.low
        self.pow = pow
        self.weight = weight
        self.eps = 0.01

    def cost(self, x):
        '''
        Returns the actual cost of the barrier
        '''
        cost = 0.
        cost += np.sum((x > self.explr_space.high-self.eps) * (x - (self.explr_space.high-self.eps))**self.pow)
        cost += np.sum((x < self.explr_space.low+self.eps) * (x - (self.explr_space.low+self.eps))**self.pow)
        return self.weight * cost

    def dx(self, x):
        '''
        Returns the derivative of the barrier wrt to the exploration
        state
        '''
        dx = np.zeros(x.shape)
        dx += 2*(x > (self.explr_space.high-self.eps)) * (x - (self.explr_space.high-self.eps))
        dx += 2*(x < (self.explr_space.low+self.eps)) * (x - (self.explr_space.low+self.eps))
        return self.weight * dx
