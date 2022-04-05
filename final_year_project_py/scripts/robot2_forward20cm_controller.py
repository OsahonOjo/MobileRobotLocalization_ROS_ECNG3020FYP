#!/usr/bin/env python

# This script continuously receives pose data for robot2 from CoppeliaSim,
# and uses that data to determine whether the robot should keep moving forward or stop moving

from __future__ import print_function

import sys, rospy, math, numpy
from std_msgs.msg import Int8
from final_year_project_py.srv import Robot2Forward40cmControllerReset, Robot2Forward40cmControllerResetResponse
from final_year_project_py.msg import PoseWithCovarianceAndUncertainty
from geometry_msgs.msg import Pose2D


WHEEL_FSM_STOP = 0
WHEEL_FSM_MOVE = 1

distance_to_target = 0.2
target_pose_calculated = False
target_pose_reached = False
target_pose = {'x': 0, 'y': 0}

command_pub = rospy.Publisher('/robot2_forward40cm_controller/position_command', Int8, queue_size=10)  # pose estimate publisher  
target_pose_pub = rospy.Publisher('/robot2_forward40cm_controller/target_pose', Pose2D, queue_size=10)


# determine target pose
def calc_target_pose(pose2D):
	target_pose = {'x': 0, 'y': 0}
	target_pose['x'] = pose2D.x + (distance_to_target * math.cos(pose2D.theta))	# distance (in meters) to target position
	target_pose['y'] = pose2D.y + (distance_to_target * math.sin(pose2D.theta))
	return target_pose


# checks if robot has reached or passes target pose
# also publishes target pose
def check_pose_reached_callback(pose2D):
	global target_pose_reached, target_pose_calculated, target_pose
	
	if target_pose_calculated == False:
		target_pose = calc_target_pose(pose2D)
		target_pose_calculated = True
	
	# the "- 0.001" is to account some ways in which target and estimated pose might not be aligned
	if pose2D.x >= (target_pose['x'] - 0.001) and pose2D.y >= (target_pose['y'] - 0.001):
		target_pose_reached = True
		return
	
	if target_pose_calculated == True and target_pose_reached == False:
		target_pose2D = Pose2D()
		target_pose2D.x = target_pose['x']
		target_pose2D.y = target_pose['y']
		target_pose2D.theta = pose2D.theta
		target_pose_pub.publish(target_pose2D)
	

# moves wheels forward if target_pose_reached is False
# stops wheels when target_pose_reached is True
def actuate_wheels_callback(event):
	global target_pose_reached
	
	int8 = Int8()
	if target_pose_reached == True:
		int8.data = WHEEL_FSM_STOP
	elif target_pose_reached == False:
		int8.data = WHEEL_FSM_MOVE
	command_pub.publish(int8)


# resets controller
def handle_reset(request):
	global target_pose_calculated, target_pose_reached, target_pose
	target_pose_calculated = False
	target_pose_reached = False
	target_pose = {'x': 0, 'y': 0}
	return Robot2Forward40cmControllerResetResponse(not target_pose_calculated)
	

def pose_listener():
	rospy.init_node('robot2_forward40cm_controller', anonymous=False)
	#rospy.Subscriber('/robot2/wheel_positions', Point, actuate_wheels_callback)
	rospy.Subscriber('/robot2/pose_estimate', PoseWithCovarianceAndUncertainty, check_pose_reached_callback)		# pose data: meters, rad.
	reset_srv = rospy.Service('robot2_forward40cm_controller_reset', Robot2Forward40cmControllerReset, handle_reset)
	timer = rospy.Timer(rospy.Duration(0.05), actuate_wheels_callback)
	rospy.spin()
	timer.shutdown()


if __name__ == '__main__':
	try:
		pose_listener()
	except rospy.ROSInterruptException:
		pass

