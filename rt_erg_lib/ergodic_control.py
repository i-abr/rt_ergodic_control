import numpy as np
from basis import Basis
from barrier import Barrier
from replay_buffer import ReplayBuffer

class RTErgodicControl(object):

    def __init__(self, model, target_dist,
                    weights=None, horizon=100, num_basis=5,
                    capacity=100000, batch_size=20):

        self.model       = model
        self.target_dist = target_dist
        self.horizon     = horizon
        self.replay_buffer = ReplayBuffer(capacity)
        self.batch_size    = batch_size

        self.basis = Basis(self.model.explr_space, num_basis=num_basis)
#         self.lamk  = 1.0/(np.linalg.norm(self.basis.k, axis=1) + 1)**(3.0/2.0)
        self.lamk = np.exp(-0.8*np.linalg.norm(self.basis.k, axis=1))
        self.barr = Barrier(self.model.explr_space)
        # initialize the control sequence
        # self.u_seq = [np.zeros(self.model.action_space.shape)
        #                 for _ in range(horizon)]
        self.u_seq = [0.0*self.model.action_space.sample()
                        for _ in range(horizon)]
        if weights is None:
            weights = {'R' : np.eye(self.model.action_space.shape[0])}

        self.Rinv = np.linalg.inv(weights['R'])

        self._phik = None
        self.ck = None

        

    def reset(self):
        self.u_seq = [0.0*self.model.action_space.sample()
                for _ in range(self.horizon)]
        self.replay_buffer.reset()

    @property
    def phik(self):
        return self._phik

    @phik.setter
    def phik(self, phik):
        assert len(phik) == self.basis.tot_num_basis, 'phik does not have the same number as ck'
        self._phik = phik


    def __call__(self, state, ck_list=None, agent_num=None):
        assert self.phik is not None, 'Forgot to set phik, use set_target_phik method'

        self.u_seq[:-1] = self.u_seq[1:]
        self.u_seq[-1]  = np.zeros(self.model.action_space.shape)

        x = self.model.reset(state)

        pred_traj = []
        dfk = []
        fdx = []
        fdu = []
        dbar= []
        for t in range(self.horizon):
            # collect all the information that is needed
            pred_traj.append(x[self.model.explr_idx])
            dfk.append(self.basis.dfk(x[self.model.explr_idx]))
            fdx.append(self.model.fdx(x, self.u_seq[t]))
            fdu.append(self.model.fdu(x))
            dbar.append(self.barr.dx(x[self.model.explr_idx]))
            # step the model forwards
            x = self.model.step(self.u_seq[t] * 0.)

        # sample any past experiences
        if len(self.replay_buffer) > self.batch_size:
            past_states = self.replay_buffer.sample(self.batch_size)
            pred_traj = pred_traj + past_states
        else:
            past_states = self.replay_buffer.sample(len(self.replay_buffer))
            pred_traj = pred_traj + past_states

        # calculate the cks for the trajectory *** this is also in the utils file
        N = len(pred_traj)
        ck = np.sum([self.basis.fk(xt) for xt in pred_traj], axis=0) / N

        self.ck = ck.copy()
        if ck_list is not None:
            ck_list[agent_num] = ck
            ck = np.mean(ck_list, axis=0)
                
        fourier_diff = self.lamk * (ck - self.phik)
        fourier_diff = fourier_diff.reshape(-1,1)

        # backwards pass
        rho = np.zeros(self.model.observation_space.shape)
        for t in reversed(range(self.horizon)):
            edx = np.zeros(self.model.observation_space.shape)
            edx[self.model.explr_idx] = np.sum(dfk[t] * fourier_diff, 0)

            bdx = np.zeros(self.model.observation_space.shape)
            bdx[self.model.explr_idx] = dbar[t]
            rho = rho - self.model.dt * (- edx - bdx - np.dot(fdx[t].T, rho))

            self.u_seq[t] = -np.dot(np.dot(self.Rinv, fdu[t].T), rho)
            if (np.abs(self.u_seq[t]) > 1.0).any():
                self.u_seq[t] /= np.linalg.norm(self.u_seq[t])

        self.replay_buffer.push(state[self.model.explr_idx])

        return self.u_seq[0].copy()
