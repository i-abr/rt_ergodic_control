#!/usr/bin/env python

##########################
##### Python Imports #####
##########################
import numpy as np
import matplotlib.pyplot as plt
import copy
import os
import argparse
import sys

##########################
###### ROS Imports #######
##########################
import rospy
import tf

##########################
### File/Local Imports ###
##########################
from rt_erg_lib.agent import Agent

class Create_Swarm():

    def __init__(self):

        rospy.init_node('swarm')
        self.num_drones = 3

        self.swarm = [None]*self.num_drones
        for i in range(self.num_drones):
            self.swarm[i] = Agent(i+1, self.num_drones)
            
        self.rate = rospy.Rate(30)

    def run(self):
        
        while not rospy.is_shutdown():

            # Control for Drones
            for i in range(self.num_drones): 
                ctrl = self.swarm[i].controller(self.swarm[i].state)
                state = self.swarm[i].step(ctrl)
                self.swarm[i].broadcast.sendTransform((self.swarm[i].state[0], self.swarm[i].state[1], 1.0),
                                             (0, 0, 0, 1), # quaternion
                                             rospy.Time.now(),
                                             self.swarm[i].agent_name,
                                             "world")
            self.rate.sleep()

    # def update_targetdist(self):
        

if __name__ == '__main__':
    swarm = Create_Swarm()

    try:
        swarm.run()
    except rospy.ROSInterruptException:
        pass
        
        
