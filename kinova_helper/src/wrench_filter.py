#!/usr/bin/env python

"""
Author: Burak Aksoy
Node: kinova_wrench_filter
Description:
    By reading tool_wrench topic represented in arm base, which has the noisy 
    current force and torque readings applied by the kinova arm robot, 
    applies a specified deadzone and a low pass filter on these raw values. Then,
    publishes the externally applied toces/torques to end effector wrt. kinova arm base
    
Motivation:
    
Parameters:
    - Topic names to subscribe and publish
    - Robot name
Subscribes to:
    - /tool_wrench_raw (geometry_msgs.msg.WrenchStamped) (wrt base)
Publishes to:
    - /tool_wrench_filtered (geometry_msgs::WrenchStamped) (wrt base)
Broadcasts to:
    - 
"""

import rospy

import geometry_msgs.msg 
# import kinova_msgs.msg

import numpy as np
import math

class KinovaWrenchFilter():
    def __init__(self):
        rospy.init_node('kinova_wrench_filter', anonymous=True)

        # Topic name to publish
        self.tool_wrench_topic_name_out = rospy.get_param("~tool_wrench_topic_name_out", "tool_wrench_filtered")
        # Publisher
        self.pub_tool_wrench_filtered = rospy.Publisher(self.tool_wrench_topic_name_out, geometry_msgs.msg.WrenchStamped, queue_size=2)

        # Topic name to subsribe
        self.tool_wrench_topic_name_in = rospy.get_param("~tool_wrench_topic_name_in", "tool_wrench_raw")
        # Subscriber
        rospy.Subscriber(self.tool_wrench_topic_name_in, geometry_msgs.msg.WrenchStamped, self.wrench_callback , queue_size=None)

        # Get deadzone and low-pass filter parameters
        self.wrench_filter_factor = rospy.get_param("~wrench_filter_factor", 0.1)
        self.force_dead_zone_thres = rospy.get_param("~force_dead_zone_thres", 10.) # N
        self.torque_dead_zone_thres = rospy.get_param("~torque_dead_zone_thres", 2.) # Nm

        # Make sure wrench_filter_factor is between 0-1
        if self.wrench_filter_factor < 0.0:
            self.wrench_filter_factor = 0.
        if self.wrench_filter_factor > 1.0:
            self.wrench_filter_factor = 1.0

        self.wrench = np.zeros(6) # variable to store previous wrench value for low-pass filter

    def wrench_callback(self,wrench_stamped_msg):
        # Take negative of the read wrench bcs we want to find the external wrench, not the one that robot applies
        wrench = -np.array([wrench_stamped_msg.wrench.torque.x, wrench_stamped_msg.wrench.torque.y, wrench_stamped_msg.wrench.torque.z,
                            wrench_stamped_msg.wrench.force.x, wrench_stamped_msg.wrench.force.y, wrench_stamped_msg.wrench.force.z])
        
        # Apply deadzones
        for i in range(3):
            if abs(wrench[i]) < self.torque_dead_zone_thres:
                wrench[i] = 0.0
            if abs(wrench[i+3]) < self.force_dead_zone_thres:
                wrench[i+3] = 0.0

        # Apply lowpass filter
        self.wrench =  (1.0-self.wrench_filter_factor)*self.wrench + self.wrench_filter_factor*wrench

        # Publish the wrench
        self.publishWrenchStamped(wrench_stamped_msg.header, self.wrench)


    def publishWrenchStamped(self, header, wrench):
        wrench_stamped_msg = geometry_msgs.msg.WrenchStamped()
        wrench_stamped_msg.header = header

        # now = rospy.Time.now()
        # rospy.loginfo("WRENCH: Added time delay %i secs %i nsecs", (now.secs - header.stamp.secs), (now.nsecs -header.stamp.nsecs))
        
        wrench_stamped_msg.wrench.force.x = wrench[3]
        wrench_stamped_msg.wrench.force.y = wrench[4]
        wrench_stamped_msg.wrench.force.z = wrench[5]
        wrench_stamped_msg.wrench.torque.x = wrench[0]
        wrench_stamped_msg.wrench.torque.y = wrench[1]
        wrench_stamped_msg.wrench.torque.z = wrench[2]

        self.pub_tool_wrench_filtered.publish(wrench_stamped_msg)


if __name__ == '__main__':
    kinovaWrenchFilter = KinovaWrenchFilter()
    rospy.spin()