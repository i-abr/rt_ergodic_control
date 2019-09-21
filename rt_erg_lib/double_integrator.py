import numpy as np
from gym.spaces import Box

class DoubleIntegrator(object):


    def __init__(self):

        self.observation_space = Box(np.array([0., 0., -np.inf, -np.inf]),
                                    np.array([1.0, 1.0, np.inf, np.inf]),
                                    dtype=np.float32)

        self.action_space = Box(np.array([-1., -1.]),
                                    np.array([1.0, 1.0]),
                                    dtype=np.float32)

        self.explr_space = Box(np.array([0., 0.]),
                                    np.array([1.0, 1.0]),
                                    dtype=np.float32)

        self.explr_idx  = [0, 1]

        self.dt = 0.1

        self.A = np.array([
                [0., 0., 1.0-0.2, 0.],
                [0., 0., 0., 1.0-0.2],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.]
        ])# - np.diag([0,0,1,1]) * 0.25

        self.B = np.array([
                [0., 0.],
                [0., 0.],
                [1.0, 0.],
                [0., 1.0]
        ])

        self.reset()


    def fdx(self, x, u):
        '''
        State linearization
        '''
        return self.A.copy()

    def fdu(self, x):
        '''
        Control linearization
        '''
        return self.B.copy()

    def reset(self, state=None):
        '''
        Resets the property self.state
        '''
        if state is None:
            self.state     = np.zeros(self.observation_space.shape[0])
            self.state[:2] = np.random.uniform(0., 0.9, size=(2,))
        else:
            self.state = state.copy()

        return self.state.copy()

    def f(self, x, u):
        '''
        Continuous time dynamics
        '''
        return np.dot(self.A, x) + np.dot(self.B, u)


    def step(self, a):
        '''
        Basic euler step
        '''
        # TODO: include ctrl clip
        self.state = self.state + self.f(self.state, a) * self.dt
        return self.state.copy()
