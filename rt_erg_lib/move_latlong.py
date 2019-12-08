#!/usr/bin/env python

import rospy
import time
import numpy as np
from geometry_msgs.msg import PoseStamped, Quaternion, PoseArray, Pose
import tf
import math
import actionlib
from move_base_msgs.msg import MoveBaseAction
from global_position_controller.srv import GoalPosition
import utm

from rt_erg_lib import Agent

class GroundVehicle(Agent):
    def __init__(self):

        Agent.__init__(self)

        rospy.init_node(self.agent_name, anonymous=True)

        self.curr_lat = None
        self.curr_lon = None
        rospy.Subscriber('latlong_pos', PoseStamped, self.latlong_callback)
        rospy.Subscriber('grid_pts', PoseArray, self.get_grid)

        self.pose_publisher = rospy.Publisher('target_pose', PoseStamped, queue_size=1)
        self.goal_pose = GoalPosition()
        self.goal_pose.target_heading = -1
        self.sendPose = rospy.ServiceProxy('goto_position', GoalPosition)

        # self.rate = rospy.Rate(10)

        #self._coord2 = None
        #self._coord1 = None

        self._coord1 = Pose()
        self._coord2 = Pose()

        self._coord1.position.x = 31.1379680
        self._coord1.position.y = -89.0647412

        self._coord2.position.x = 31.137823
        self._coord2.position.y = -89.064585


        self._delta_x = self._coord2.position.x-self._coord1.position.x
        self._delta_y = self._coord2.position.y- self._coord1.position.y


        self.state = np.array([0.,0.,0.,0.])
        self._t_theta = 0.

    def _coord_to_dist(self, coord):
        x = coord.pose.position.x/self._delta_x
        y = coord.pose.position.y/self._delta_y
        return [x, y]

    def _dist_to_coord(self, dist):
        lat = dist[0] * self._delta_x + self._coord1.position.x
        lon = dist[1] * self._delta_y + self._coord1.position.y
        return [lat, lon]

    def get_grid(self, msg):
        print(msg)
        self._coord1 = msg.poses[0]
        self._coord2 = msg.poses[1]
        self._delta_x = self._coord2.position.x-self._coord1.position.x
        self._delta_y = self._coord2.position.y-self._coord1.position.y


    def latlong_callback(self, data):
        x, y = self._coord_to_dist(data)
        # TODO: figure out this interface
        # self.state[0] = x
        # self.state[1] = y

    def step(self):
        if (self._coord1 is None) or self._coord2 is None:
            pass
        else:
            next_pose = self.control_step()
            x = next_pose[0] #.5#*np.cos(self._t_theta) + 0.5
            y = next_pose[1] #0.#*np.sin(self._t_theta) + 0.5
            coord = self._dist_to_coord([x, y])
            self.goal_pose.target_latitude = coord[0]
            self.goal_pose.target_longitude = coord[1]
            try:
                print("Sending pose")
                print("(%f,%f)" % (coord[0], coord[1]))
                self.sendPose(self.goal_pose.target_latitude, self.goal_pose.target_longitude, -1)
            except rospy.ServiceException, e:
                print "Service call failed: %s"%e


if __name__ == '__main__':
    rospy.wait_for_service('goto_position')
    print("Got the goto_position service.")
    gv = GroundVehicle()
    rate = rospy.Rate(1)
    try:
        while not rospy.is_shutdown():
            gv.step()
            rate.sleep()
    except rospy.ROSInterruptException as e:
        print('clean break')
