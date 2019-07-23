import numpy as np
from numpy import pi, cos, sin

class Basis(object):
    '''
    Cosine basis functions for decomposing distributions
    '''
    def __init__(self, explr_space, num_basis=5, offset=None):
        if offset is not None:
            raise NotImplementedError('Have not implemented offsets')
        self.dl  = explr_space.high - explr_space.low

        n = explr_space.shape[0]
        k = np.meshgrid(*[[i for i in range(num_basis)] for _ in range(n)])

        self.k = np.c_[k[0].ravel(), k[1].ravel()]

        self.hk = np.zeros(self.k.shape[0])

        for i, k in enumerate(self.k):
            if np.prod(k) < 1e-5:
                self.hk[i] = 1.
            else:
                top = np.prod(self.dl * (2.0 * k * np.pi + np.sin(2.0 * k *np.pi)) )
                bot = 16.0 * np.prod(k) * np.pi**2
                self.hk[i] = top/bot

        self.tot_num_basis = num_basis**n

    def fk(self, x):
        assert (x.shape[0] == self.dl.shape[0]), 'input dim does not match explr dim'
        return np.prod(np.cos(np.pi*x/self.dl * self.k),1)#/self.hk

    def dfk(self, x):
         dx = np.zeros((self.tot_num_basis, x.shape[0]))
         dx[:,0] = -self.k[:,0]*pi*sin(pi * self.k[:,0] * x[0]/self.dl[0]) * cos(pi * self.k[:,1]*x[1]/self.dl[1])#/self.hk
         dx[:,1] = -self.k[:,1]*pi*sin(pi * self.k[:,1] * x[1]/self.dl[1]) * cos(pi * self.k[:,0]*x[0]/self.dl[0])#/self.hk
         return dx
