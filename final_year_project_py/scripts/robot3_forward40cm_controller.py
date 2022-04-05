#!/usr/bin/env python

# This script continuously receives pose data for robot3 from CoppeliaSim,
# and uses that data to determine whether the robot should keep moving forward or stop moving

from __future__ import print_function

import sys, rospy, math, numpy
from final_year_project_py.srv import Robot3Forward40cmControllerReset, Robot3Forward40cmControllerResetResponse
from final_year_project_py.msg import PoseWithCovarianceAndUncertainty  #change here
from std_msgs.msg import Float64  #change here
from geometry_msgs.msg import Pose2D


distance_to_target = 0.4
target_pose_calculated = False
target_pose = {'x': 0, 'y': 0}

motion_command = 0
controller = {'pid_kp': math.pi/distance_to_target, 'pid_ki': 0, 'pid_kd': 0, 'pid_output': 0,  'pid_output_max': math.pi, 'pid_output_min': 0, 'pid_proportional': 0, 'pid_integral': 0, 'pid_error_prev': 0, 'pid_integral_max': 0.5, 'pid_integral_min': 0, 'pid_derivative': 0}
CONTROLLER_PERIOD = 0.5		# because dead-reckoning node for robot 3 publishes every 0.5s

command_pub = rospy.Publisher('/robot3_forward40cm_controller/motion_command', Float64, queue_size=10)  # pose estimate publisher  
target_pose_pub = rospy.Publisher('/robot3_forward40cm_controller/target_pose', Pose2D, queue_size=10)


def controller_callback(pose2D):  #change here
	global target_pose_calculated, target_pose
	
	if target_pose_calculated == False:
		target_pose = calc_target_pose(pose2D)
		target_pose_calculated = True
		
	target_pose2D = Pose2D()
	target_pose2D.x = target_pose['x']
	target_pose2D.y = target_pose['y']
	target_pose2D.theta = pose2D.theta
	target_pose_pub.publish(target_pose2D)
	
	motion_command = Float64()  #several changes
	motion_command.data = 0
	
	if pose2D.x >= target_pose['x'] and pose2D.y >= target_pose['y']:	# if pose.x or pose.y exceeds target.x or target.y, stop the robot
		motion_command.data = 0
	else:		
		pid_motion_controller(pose2D)
		motion_command.data = controller['pid_output']
		
	command_pub.publish(motion_command)


def calc_distance(pose2D):
	global target_pose
	x_distance = target_pose['x'] - pose2D.x
	y_distance = target_pose['y'] - pose2D.y
	distance = math.sqrt((x_distance**2) + (y_distance**2))
	return distance


def calc_error(pose2D):
	global target_pose
	distance = calc_distance(pose2D)
	
	if pose2D.x >= target_pose['x'] and pose2D.y >= target_pose['y']:	# if pose.x or pose.y exceeds target.x or target.y
		distance = -1 * distance
	return distance


def calc_target_pose(pose2D):
	target_pose = {'x': 0, 'y': 0}
	target_pose['x'] = pose2D.x + (distance_to_target * math.cos(pose2D.theta))	# 0.5 as in 0.5m = 50cm
	target_pose['y'] = pose2D.y + (distance_to_target * math.sin(pose2D.theta))
	return target_pose


def pid_motion_controller(pose2D):
	pid_error = calc_error(pose2D)
	
	controller['pid_proportional'] = controller['pid_kp'] * pid_error
	controller['pid_integral'] = controller['pid_integral'] + ((0.5 * controller['pid_ki'] * CONTROLLER_PERIOD) * (pid_error + controller['pid_error_prev']))

	#integral anti-windup by clamping integral controller output
	if controller['pid_integral'] > controller['pid_integral_max']:
		controller['pid_integral'] = controller['pid_integral_max']
	elif controller['pid_integral'] < controller['pid_integral_min']:
		controller['pid_integral'] = controller['pid_integral_min']

	controller['pid_output'] = controller['pid_proportional'] + controller['pid_integral'] + controller['pid_derivative']

	#pid controller output clamping
	if controller['pid_output'] > controller['pid_output_max']:
		controller['pid_output'] = controller['pid_output_max']
	elif controller['pid_output'] < controller['pid_output_min']:
		controller['pid_output'] = controller['pid_output_min']

	controller['pid_error_prev'] = pid_error


def handle_reset(request):
	global target_pose_calculated, target_pose, motion_command, controller
	target_pose_calculated = False
	target_pose = {'x': 0, 'y': 0}
	motion_command = 0
	controller['pid_output'] = 0
	controller['pid_proportional'] = 0
	return Robot3Forward40cmControllerResetResponse(not target_pose_calculated)


def pose_listener():
	rospy.init_node('robot3_forward40cm_controller', anonymous=False)
	rospy.Subscriber('/robot3/pose_estimate', PoseWithCovarianceAndUncertainty, controller_callback) #change here			# pose data: meters, rad.
	reset_srv = rospy.Service('robot3_forward40cm_controller_reset', Robot3Forward40cmControllerReset, handle_reset)
	rospy.spin()


if __name__ == '__main__':
	try:
		pose_listener()
	except rospy.ROSInterruptException:
		pass

