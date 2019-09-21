from double_integrator import DoubleIntegrator
from ergodic_control import RTErgodicControl
from target_dist import TargetDist
from utils import convert_phi2phik, convert_ck2dist, convert_traj2ck, convert_phik2phi
import numpy as np

import matplotlib.pyplot as plt

import os

import rospy
import tf

from rt_ergodic_control.msg import Ck_msg
from vr_exp_ros.msg import Target_dist
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point

class Agent(DoubleIntegrator):

    def __init__(self, agent_num=0, tot_agents=1):

        DoubleIntegrator.__init__(self)

        rospy.init_node('agent{}'.format(agent_num))
        self.rate = rospy.Rate(30)

        self.agent_num = agent_num
        self.tot_agents = tot_agents
        self.agent_name = 'agent{}'.format(agent_num)
        self.model      = DoubleIntegrator()

        self.t_dist      = TargetDist(num_nodes=3) #TODO: remove example target distribution
        
        self.controller  = RTErgodicControl(self.model, self.t_dist,
                                            horizon=15, num_basis=5, batch_size=200,capacity=500)#, batch_size=-1)

        # setting the phik on the ergodic controller
        self.controller.phik = convert_phi2phik(self.controller.basis,
                                                self.t_dist.grid_vals,
                                                self.t_dist.grid)
        self.tdist_sub = rospy.Subscriber('/target_distribution', Target_dist, self.update_tdist)

        self.reset() # reset the agent

        self.broadcast = tf.TransformBroadcaster()
        self.broadcast.sendTransform((self.state[0], self.state[1], 1.0),
                                     (0, 0, 0, 1), # quaternion
                                     rospy.Time.now(),
                                     self.agent_name,
                                     "world")
        self.__max_points = 1000
        self.marker_pub = rospy.Publisher('/agent{}/marker_pose'.format(agent_num), Marker, queue_size=1)
        self.marker = Marker()
        self.marker.id = agent_num
        self.marker.type = Marker.LINE_STRIP
        self.marker.header.frame_id = "world"
        self.marker.scale.x = .01
        self.marker.scale.y = .01
        self.marker.scale.z = .01

        color = [0.0,0.0,0.0]
        color[agent_num] = 1.0
        self.marker.color.a = .7
        self.marker.color.r = color[0]
        self.marker.color.g = color[1]
        self.marker.color.b = color[2]

        # Publisher for Ck Sharing
        self.ck_pub = rospy.Publisher('agent{}/ck'.format(agent_num), Ck_msg, queue_size=1)

        # Ck Subscriber for all agent ck's

        self.ck_list = [None]*tot_agents
        for i in range(tot_agents):
            rospy.Subscriber('agent{}/ck'.format(i), Ck_msg, self.update_ck)
            

    def update_ck(self, data):
        i = data.agent_num
        self.ck_list[i] = np.array(data.ck_array)        

    def update_tdist(self, data):
        print("updating tdist in subscriber")
        self.t_dist.grid_vals = np.array(data.target_array)
        print("t_dist shape: ", self.t_dist.grid_vals.shape)
        self.t_dist.has_update = True
        
    def run(self):
        while not rospy.is_shutdown():

            # If necessary, update target distribution:
            if self.t_dist.has_update == True:
                self.controller.phik = convert_phi2phik(self.controller.basis, self.t_dist.grid_vals)
                self.t_dist.has_update = False

            comm_link = True
            for _ck in self.ck_list:
                if _ck is None:
                    comm_link = False
            # Update ck in controller
            if comm_link:
                ctrl = self.controller(self.state, self.ck_list, self.agent_num)
            else:
                ctrl = self.controller(self.state)

            # Publish Current Ck
            ck_msg = Ck_msg()
            ck_msg.agent_num = self.agent_num
            ck_msg.ck_array = self.controller.ck.copy()
            self.ck_pub.publish(ck_msg)
            
            state = self.step(ctrl)

            
            self.broadcast.sendTransform((state[0], state[1], 1.0),
                                         (0, 0, 0, 1), # quaternion
                                         rospy.Time.now(),
                                         self.agent_name,
                                         "world")
            pnt = Point()
            pnt.x = state[0]
            pnt.y = state[1]
            pnt.z = 1.0
            if len(self.marker.points) < self.__max_points:
                self.marker.points.append(pnt)
            else:
                self.marker.points[:-1] = self.marker.points[1:]
                self.marker.points[-1] = pnt
            self.marker.pose.position.x = 0*state[0]
            self.marker.pose.position.y = 0*state[1]
            self.marker.pose.position.z = 0*1.0
            self.marker_pub.publish(self.marker)
            self.rate.sleep()
