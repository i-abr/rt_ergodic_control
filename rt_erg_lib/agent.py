from double_integrator import DoubleIntegrator
from ergodic_control import RTErgodicControl
from target_dist import TargetDist
from utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi
import numpy as np

import matplotlib.pyplot as plt

import os

import rospy
import tf

class Agent(DoubleIntegrator):

    def __init__(self, agent_num=0, tot_agents=1):

        DoubleIntegrator.__init__(self)

        rospy.init_node('agent{}'.format(agent_num))
        self.rate = rospy.Rate(30)

        self.agent_name = 'agent{}'.format(agent_num)
        self.model      = DoubleIntegrator()

        # self.tdist_sub = rospy.Subscriber('/target_distribution', Target_dist, self.update_tdist)
        # self.tdist.has_update = False 
        self.t_dist      = TargetDist(num_nodes=2) #TODO: remove example target distribution
        
        self.controller  = RTErgodicControl(self.model, self.t_dist,
                                horizon=15, num_basis=5, batch_size=-1)

        # setting the phik on the ergodic controller
        self.controller.phik = convert_phi2phik(self.controller.basis,
                                                self.t_dist.grid_vals,
                                                self.t_dist.grid)
        self.reset() # reset the agent

        self.broadcast = tf.TransformBroadcaster()
        self.broadcast.sendTransform((self.state[0], self.state[1], 1.0),
                                     (0, 0, 0, 1), # quaternion
                                     rospy.Time.now(),
                                     self.agent_name,
                                     "world")

    def update_tdist(self, data):
        self.t_dist.grid_vals = data.target_array
        num_pts = np.shape(self.t_dist.grid_vals)[0]
        grid = np.meshgrid(*[np.linspace(0, 1, num_pts) for _ in range(2)])
        self.t_dist.grid = np.c_[grid[0].ravel(), grid[1].ravel()]

        self.tdist.has_update = True
    def run(self):
        while not rospy.is_shutdown():

            # TODO:
            # if self.tdist.has_update == True:
            #     self.controller.phik = convert_phi2phik(self.controller.basis, self.t_dist.grid_vals, self.t_dist.grid)
            #     self.t_dist.has_update = False
            ctrl = self.controller(self.state)
            state = self.step(ctrl)

            self.broadcast.sendTransform((state[0], state[1], 1.0),
                                         (0, 0, 0, 1), # quaternion
                                         rospy.Time.now(),
                                         self.agent_name,
                                         "world")
            self.rate.sleep()
