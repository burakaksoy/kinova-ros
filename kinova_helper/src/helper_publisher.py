#!/usr/bin/env python

"""
Author: Burak Aksoy
Node: kinova_helper_publisher
Description:
    By reading joint_states topic, which has the current joint angles of the 
    kinova arm and joint efforts as the torques applied by the joints of the 
    kinova arm, calculates and publishes the effective end effector forces (wrench)
    and the end effector pose.
Motivation:
    
Parameters:
    - The j2n6s300 kinova arm's link lengths
    - Topic names to subscribe and publish
    - Robot name
Subscribes to:
    - /j2n6s300_driver/out/joint_state (sensor_msgs::JointState)
Publishes to:
    - /j2n6s300_tool_wrench (geometry_msgs::WrenchStamped) (wrt base)
    - /j2n6s300_tool_pose (geometry_msgs::PoseStamped) (wrt base)
Broadcasts to:
    - 
"""

import rospy

import numpy as np

import sensor_msgs.msg
import geometry_msgs.msg 
import std_msgs.msg
# from geometry_msgs.msg import WrenchStamped

# import kinova_msgs.msg

# To create kinova robot jacobian
import general_robotics_toolbox as rox
import math
# from math import pi, sin,cos,atan2

import tf.transformations

class KinovaHelperPublisher():
    def __init__(self):
        rospy.init_node('kinova_helper_publisher', anonymous=True)

        self.kinova_robotName = rospy.get_param("~robot_name", "j2n6s300")
        self.tf_prefix_ = self.kinova_robotName + "_"

        # Topic name to publish
        self.tool_wrench_topic_name = rospy.get_param("~tool_wrench_topic_name", "out_custom/tool_wrench")
        self.tool_pose_topic_name = rospy.get_param("~tool_pose_topic_name", "out_custom/tool_pose")

        # Publisher
        self.pub_tool_wrench_stamped = rospy.Publisher(self.tool_wrench_topic_name, geometry_msgs.msg.WrenchStamped, queue_size=2)
        self.pub_tool_pose_stamped = rospy.Publisher(self.tool_pose_topic_name, geometry_msgs.msg.PoseStamped, queue_size=2)
        # Minimum singular value publisher for debug
        self.min_singular_value_topic_name = rospy.get_param("~min_singular_value_topic_name", "min_singular_value")
        self.pub_min_singular_value = rospy.Publisher(self.min_singular_value_topic_name, std_msgs.msg.Float64, queue_size=2)
        self.min_singular_value_thres = rospy.get_param("~min_singular_value_thres", 0.8)
        self.joint_torque_dead_zone_thres = rospy.get_param("~joint_torque_dead_zone_thres", 0.5) # Nm

        # Topic name to subsribe
        self.joint_state_topic_name = rospy.get_param("~joint_state_topic_name", self.kinova_robotName + "_driver/out/joint_state")

        # Subscriber
        rospy.Subscriber(self.joint_state_topic_name, sensor_msgs.msg.JointState, self.joint_state_callback , queue_size=None)

        # Kinova arm link lengths
        self.D1 = rospy.get_param("~D1", 0.2755)
        self.D2 = rospy.get_param("~D2", 0.41)
        self.e2 = rospy.get_param("~e2", 0.0098)
        self.D3 = rospy.get_param("~D3", 0.2073)
        self.D4 = rospy.get_param("~D4", 0.0741)
        self.D5 = rospy.get_param("~D5", 0.0741)
        self.D6 = rospy.get_param("~D6", 0.1600)

        # Auxiliary variables
        self.aa = math.pi/6
        self.ca = math.cos(self.aa)
        self.sa = math.sin(self.aa)
        self.c2a = math.cos(2*self.aa)
        self.s2a = math.sin(2*self.aa)
        self.d4b = self.D3 + self.D5*(self.sa/self.s2a) 
        self.d5b = self.D4*(self.sa/self.s2a) + self.D5*(self.sa/self.s2a) 
        self.d6b = self.D5*(self.sa/self.s2a) + self.D6
        
        self.ex = np.array([1,0,0])
        self.ey = np.array([0,1,0])
        self.ez = np.array([0,0,1])

        # Product of exponential parameters
        h1 = -self.ez
        h2 = self.ey
        h3 = -self.ey
        h4 = self.ez
        h5 = -self.ey*self.ca + self.ez*self.sa
        h6 = -self.ey*self.ca - self.ez*self.sa
        self.H = np.array([h1, h2, h3, h4, h5, h6]).T

        P01 = (self.D1)*self.ez
        P12 = 0*self.ez
        P23 = (self.D2)*self.ex + (self.e2)*self.ey
        P34 = 0*self.ez
        P45 = (self.d5b*self.ca)*self.ey - (self.d5b*self.sa + self.d4b)*self.ez
        P56 = 0*self.ez 
        P6e = (self.d6b*self.ca)*self.ey + (self.d6b*self.sa)*self.ez 
        self.P = np.array([P01, P12, P23, P34, P45, P56, P6e]).T

        # Tool frame (end effector) adjustment 
        self.p_tool = 0*self.ez
        ex_tool = -self.ey*self.sa + self.ez*self.ca
        ey_tool = self.ex
        ez_tool = self.ey*self.ca + self.ez*self.sa
        self.R_tool = np.array([ex_tool,ey_tool,ez_tool]).T

        # Parameters to create the robot object properly
        self.joint_types = np.array([0,0,0,0,0,0]) # All revolute
        # self.joint_upper_limits = np.radians([10000,310,341,10000,10000,10000])
        # self.joint_lower_limits = np.radians([-10000,50,19,-10000,-10000,-10000])
        self.joint_upper_limits = None
        self.joint_lower_limits = None

        # Joint angles in Zero config 
        self.q_zero = np.deg2rad(np.array([180,270,90,180,180,0]))

        # Create the kinova robot object with the general robotics toolbox
        self.kinova = rox.Robot(self.H,
                                self.P,
                                self.joint_types,
                                self.joint_lower_limits,
                                self.joint_upper_limits, 
                                R_tool=self.R_tool, p_tool=self.p_tool)



    def joint_state_callback(self,joint_state_msg):
        joint_state_msg.name # list of joint names, 12 elements, last 6 are for the fingers
        
        # Get current joint angles (rad)
        q = np.array(joint_state_msg.position[:6]) - self.q_zero # joint angles in radian (12 elements, 6 are needed)
        # Get current joint velocities (rad/s)
        q_dot = np.array(joint_state_msg.velocity[:6]) # joint velocity in radian/s (12 elements, 6 are needed)
        # Get current joint torques (Nm)
        tau = np.array(joint_state_msg.effort[:6]) # joint torques in Nm (12 elements, 6 are needed)

        # Apply deadzones for joint torques
        for i in range(6):
            if abs(tau[i]) < self.joint_torque_dead_zone_thres:
                tau[i] = 0.0

        # Calculate the current Jacobian of end effector wrt base
        J = rox.robotjacobian(self.kinova,q)
        # rospy.loginfo("Current Jacobian: " + str(J))

        # put a warning if the jacobian is near singularity
        min_sing_val = np.linalg.svd(J, compute_uv=False)[-1] # Minimum singular value of the jacobian
        min_sing_val_msg = std_msgs.msg.Float64()
        min_sing_val_msg.data = min_sing_val
        self.pub_min_singular_value.publish(min_sing_val_msg)
        if min_sing_val <= self.min_singular_value_thres:
            rospy.logwarn("Jacobian minimum signular value is less than the threshold value: " + str(self.min_singular_value_thres) + "!!!")

        # Calculate the current end effector forces (wrench) wrt base
        Ftip = np.linalg.pinv(J.T).dot(tau)
        # rospy.loginfo("Current F_tip [Tau;F]: " + str(Ftip))
        self.publishWrenchStamped(joint_state_msg.header, Ftip)

        # Calculate current end effector pose wrt base
        T = rox.fwdkin(self.kinova,q)
        # rospy.loginfo("Current End Effector position in base frame [P|R]:\n" + str(T))
        # rospy.loginfo("Current End Effector orientation XYZ euler angles in base frame [z,y,x]: " + str(tf.transformations.euler_from_matrix(T.R,axes='szyx')))
        self.publishPoseStamped(joint_state_msg.header, T.p, rox.R2q(T.R))

        # # Calculate current end effector velocity wrt base
        # V = J.dot(q_dot)
        # rospy.loginfo("Current End Effector velocity in base frame [w;v]: " + str(V))



    def publishWrenchStamped(self, header, wrench):
        wrench_stamped_msg = geometry_msgs.msg.WrenchStamped()
        wrench_stamped_msg.header = header
        wrench_stamped_msg.header.frame_id = self.tf_prefix_ + "link_base"

        # now = rospy.Time.now()
        # rospy.loginfo("WRENCH: Added time delay %i secs %i nsecs", (now.secs - header.stamp.secs), (now.nsecs -header.stamp.nsecs))
        
        wrench_stamped_msg.wrench.force.x = wrench[3]
        wrench_stamped_msg.wrench.force.y = wrench[4]
        wrench_stamped_msg.wrench.force.z = wrench[5]
        wrench_stamped_msg.wrench.torque.x = wrench[0]
        wrench_stamped_msg.wrench.torque.y = wrench[1]
        wrench_stamped_msg.wrench.torque.z = wrench[2]

        self.pub_tool_wrench_stamped.publish(wrench_stamped_msg)

    def publishPoseStamped(self, header, position, orientation):
        pose_stamped_msg = geometry_msgs.msg.PoseStamped()
        pose_stamped_msg.header = header
        pose_stamped_msg.header.frame_id = self.tf_prefix_ + "link_base"

        pose_stamped_msg.pose.position.x = position[0]
        pose_stamped_msg.pose.position.y = position[1]
        pose_stamped_msg.pose.position.z = position[2]
        
        pose_stamped_msg.pose.orientation.w = orientation[0]
        pose_stamped_msg.pose.orientation.x = orientation[1]
        pose_stamped_msg.pose.orientation.y = orientation[2]
        pose_stamped_msg.pose.orientation.z = orientation[3]
        

        self.pub_tool_pose_stamped.publish(pose_stamped_msg)



if __name__ == '__main__':
    kinovaHelperPublisher = KinovaHelperPublisher()
    rospy.spin()