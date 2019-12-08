from single_integrator import SingleIntegrator
from ergodic_control import RTErgodicControl
from target_dist import TargetDist
from utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi
import numpy as np


class Agent(SingleIntegrator):

    def __init__(self, agent_num=0):


        SingleIntegrator.__init__(self)
        self.agent_num  = agent_num
        self.agent_name = 'agent{}'.format(agent_num)

        self.model = SingleIntegrator()
        self.t_dist = TargetDist()

        self.controller = RTErgodicControl(self.model, self.t_dist,
                                horizon=15, num_basis=10, batch_size=100)


        self.controller.phik = convert_phi2phik(self.controller.basis,
                                                self.t_dist.grid_vals,
                                                self.t_dist.grid)
        self.reset() # need to override this

    def control_step(self):
        ctrl = self.controller(self.state)
        return self.step(ctrl)
