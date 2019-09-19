#!/usr/bin/env python


import rospy
import roslaunch 


if __name__ == '__main__':
    package = 'rt_ergodic_control'
    executable = 'create_agent.py'

    nodes       = []
    processes   = []
    launch = roslaunch.scriptapi.ROSLaunch()
    launch.start()


    num_agents = 3
    for i in range(num_agents):
        node_name = 'agent{}'.format(i)
        args = '{} {}'.format(i, num_agents)
        nodes.append(
            roslaunch.core.Node(package=package, node_type=executable, name=node_name, args=args,output="screen")
        )
        processes.append(launch.launch(nodes[-1]))

    try:
        rospy.init_node('test', anonymous=True)
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    
