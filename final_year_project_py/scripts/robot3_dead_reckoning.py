#!/usr/bin/env python

# This script continuously receives encoder ticks from CoppeliaSim, uses the ticks for dead-reckoning,
# and twice per second, publishes the current pose estimate

from __future__ import print_function

import sys, rospy, math, numpy
from std_msgs.msg import Float64MultiArray
from final_year_project_py.msg import PoseWithCovarianceAndUncertainty  


WHEEL_RADIUS = 0.035                                # wheel radius in meters
ENCODER_RESOLUTION = 250                            # 250 ticks/rotation
TICK_ANGLE = (2*math.pi) / ENCODER_RESOLUTION       # angle per tick in radians
TICK_DISTANCE = WHEEL_RADIUS * TICK_ANGLE

b = 0.1                # distance between wheels meters
kr = 0.1                 # error constant for right wheel       
kl = 0.1                 # error constant for left wheel
uncertainty = 0

pose_pub = rospy.Publisher('/robot3/pose_estimate', PoseWithCovarianceAndUncertainty, queue_size=10)  # pose estimate publisher  

state = numpy.array([[0.2498], [0.5591], [1.5706]])                   # x (m), y (m), theta (rad.)
p = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])   # state uncertainty; state is initially known so uncertainty is zero


def dead_reckon(del_sl, del_sr):
	global uncertainty, state, p
	
	del_s = (del_sr + del_sl) / 2
	del_theta = (del_sr - del_sl) / b
	theta_new = state[2][0] + (del_theta/2)

	v = numpy.array([[kr*abs(del_sr), 0], [0, kl*abs(del_sl)]])

	fp = numpy.array([[1, 0, -1*del_s*math.sin(theta_new)], [0, 1, del_s*math.cos(theta_new)], [0, 0, 1]])
	fpt = fp.transpose() 

	fs = numpy.array([[(0.5*math.cos(theta_new))-(del_s/(2*b))*math.sin(theta_new), (0.5*math.cos(theta_new))+((del_s/(2*b))*math.sin(theta_new))], [(0.5*math.sin(theta_new))+(del_s/(2*b))*math.cos(theta_new), (0.5*math.sin(theta_new))-((del_s/(2*b))*math.cos(theta_new))], [(1/b), (-1/b)]])
	fst = fs.transpose()

	state = state + [[del_s*math.cos(theta_new)], [del_s*math.sin(theta_new)], [del_theta]]

	p = fp.dot(p).dot(fpt) + fs.dot(v).dot(fst)

	pdet =  ( p[0][0] * ( (p[1][1]*p[2][2]) - (p[2][1]*p[1][2]) )) - ( p[0][1] * ( (p[1][0]*p[2][2]) - (p[2][0]*p[1][2]) )) + ( p[0][2] * ( (p[1][0]*p[2][1]) - (p[1][1]*p[2][0]) ))  
	
	uncertainty = math.sqrt(abs(pdet))


def encoder_data_array_callback(encoder_data_array):
	del_sl = encoder_data_array.data[0] * TICK_DISTANCE		# distance moved by left wheel in meters
	del_sr = encoder_data_array.data[1] * TICK_DISTANCE		# distance moved by right wheel in meters
	dead_reckon(del_sl, del_sr)


def timer_callback(event):
	pose = PoseWithCovarianceAndUncertainty()

	pose.x = round(state[0][0], 3) 
	pose.y = round(state[1][0], 3)
	pose.theta = round(state[2][0], 4)
	pose.covar = [p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2], p[2][0], p[2][1], p[2][2]]
	pose.uncertainty = round(uncertainty, 4)

	pose_pub.publish(pose)	


def listener():
	rospy.init_node('robot3_dead_reckoning', anonymous=False)
	rospy.Subscriber('/robot3/encoder_data_array', Float64MultiArray, encoder_data_array_callback)
	timer = rospy.Timer(rospy.Duration(0.5), timer_callback)            # timer to publish pose estimate at 2Hz
	rospy.spin()    
	timer.shutdown()


if __name__ == '__main__':
	try:
		listener()
	except rospy.ROSInterruptException:
		pass

