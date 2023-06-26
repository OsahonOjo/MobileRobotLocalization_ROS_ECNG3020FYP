#!/usr/bin/env python

# This script:
# 1. receives encoder data from CoppeliaSim, and
#    uses it for pose and covariance estimation (dead-reckoning),
# 2. publishes the current pose estimate and uncertainty,
#    twice per second

# python3 /home/osahon/MATLAB_WS/HoughLocalization/houghLocalization/for_redistribution_files_only/houghLocalizationSample1.py
# linesMatrix(i,:) = [lines(i).point1(1) lines(i).point1(2) lines(i).point2(1) lines(i).point2(2) lines(i).theta lines(i).rho];

# sensors with uncertainty: 
# encoders: resolution implemented in CoppeliaSim
# gyroscope: 0.05 deg/s
# accelerometer: 0.01g = 0.0981 m/s^2


from __future__ import print_function
import json
import sys, rospy, math, numpy
from std_msgs.msg import Float64MultiArray
from geometry_msgs.msg import Point, PointStamped, PoseWithCovarianceStamped
# from final_year_project_py.msg import PoseWithCovarianceAndUncertainty


# constants
WHEEL_RADIUS = 0.035   			# in meters
WHEELBASE = 0.1					# distance between wheels (meters)
PUBLISHER_TIMER_DURATION = 0.5	# seconds

# for encoder
PPR = 11                        # 11 pulses (22 transitions) per rotation
ANGLE_PER_TICK = (2*math.pi)/(PPR*2)	# 1 tick per transition

# for equations
kr = 0.1                 		# error constant: right wheel       
kl = 0.1                 		# error constant: left wheel

previousLeftWheelTicks = 0
previousRightWheelTicks = 0

initialPose2D = {}
beaconPositions = {}
with open('initialValues.json') as json_file:
    data = json.load(json_file)
    initialPose2D = data.EncodersCubeBotInitialPose
    beaconPositions = { "beacon1": data.Beacon1Position, "beacon2": data.Beacon2Position }

state = numpy.array([[initialPose2D.x], [initialPose2D.y], [initialPose2D.theta]])
p = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
uncertainty = 0

# posePublisher = rospy.Publisher('/EncodersCubeBot/pose_estimate', PoseWithCovarianceAndUncertainty, queue_size=10)  
posePublisher = rospy.Publisher('/EncodersCubeBot/pose_estimate', PoseWithCovarianceStamped, queue_size=10)  


# odometry = { x: total left wheel ticks, y: total right wheel ticks, z }
# NOTE: tightly-coupled function: modifies 5, uses several global variables
def LKMPredictionStep(odometry):
	print(odometry)

	# global keyword allows you to modify a global variable
	global previousLeftWheelTicks, previousRightWheelTicks, state, p, uncertainty

	# 1. estimate state

	leftTicksDifference = odometry.x - previousLeftWheelTicks
	rightTicksDifference = odometry.y - previousRightWheelTicks
	
	# distance s = radius * theta [meters]
	del_sl = WHEEL_RADIUS * (leftTicksDifference * ANGLE_PER_TICK)
	del_sr = WHEEL_RADIUS * (rightTicksDifference * ANGLE_PER_TICK)
	del_s = (del_sr + del_sl) / 2
	del_theta = (del_sr - del_sl) / WHEELBASE
	theta = state[2][0]
	theta_new = theta + (del_theta/2)	

	# TODO: add random Gaussian odometry noise HERE
	state = state + [[del_s*math.cos(theta_new)], [del_s*math.sin(theta_new)], [del_theta]]

	# The errors are proportional to the absolute value of the traveled distances
	odometry_covariance = numpy.array([[kr*abs(del_sr), 0], [0, kl*abs(del_sl)]])	


	# 2. predict covariance

	# partial derivative of configuration w.r.t state (x,y,z)
	fp = numpy.array([[1, 0, -1*del_s*math.sin(theta_new)], [0, 1, del_s*math.cos(theta_new)], [0, 0, 1]])
	fpt = fp.transpose()

	# partial derivative of configuration w.r.t odometry (del_sl, del_sr)
	fs = numpy.array([[(0.5*math.cos(theta_new))-(del_s/(2*WHEELBASE))*math.sin(theta_new), (0.5*math.cos(theta_new))+((del_s/(2*WHEELBASE))*math.sin(theta_new))], [(0.5*math.sin(theta_new))+(del_s/(2*WHEELBASE))*math.cos(theta_new), (0.5*math.sin(theta_new))-((del_s/(2*WHEELBASE))*math.cos(theta_new))], [(1/WHEELBASE), (-1/WHEELBASE)]])
	fst = fs.transpose()
	
	p = fp.dot(p).dot(fpt) + fs.dot(odometry_covariance).dot(fst)

	# determinant of just the state estimate part of the covariance matrix
	pdeterminant = ( p[0][0] * ( (p[1][1]*p[2][2]) - (p[2][1]*p[1][2]) )) - ( p[0][1] * ( (p[1][0]*p[2][2]) - (p[2][0]*p[1][2]) )) + ( p[0][2] * ( (p[1][0]*p[2][1]) - (p[1][1]*p[2][0]) ))
	uncertainty = math.sqrt(abs(pdeterminant))

	previousLeftWheelTicks = odometry.x
	previousRightWheelTicks = odometry.y


# beaconData: { header={ stamp={ secs, nsecs }, frame_id }, point={ x (bearing [degrees]), y, z } })
def LKMUpdateStepWithBeaconData(beaconData):
	print(beaconData)

	global state, p, uncertainty

	# 3. calculate innovation

	knownBeaconPosition = beaconPositions[beaconData.header.frame_id]	# {x,y,theta}
	robotX = state[0][0], robotY = state[1][0]
	measuredBeaconBearing = numpy.array([[0], [beaconData.point.x]])	# radians
	expectedBeaconBearing = numpy.array([[0], [math.atan2((knownBeaconPosition.y - robotY) / (knownBeaconPosition.x - robotX))]])
	
	# TODO: add random Gaussian sensor noise HERE
	innovation = measuredBeaconBearing - expectedBeaconBearing


	# 4. update state estimate

	# TODO: add random Gaussian sensor noise HERE
	sensor_covariance = numpy.array([[1, 0], [0, 1]])

	Hw = numpy.array([[1, 0], [0, 1]])
	# Hx = numpy.array([[(-()/), (), 0], [(), (), -1]])


	# 5. update covariance

	# p = fp.dot(p).dot(fpt) + fs.dot(odometry_covariance).dot(fst)

	# determinant of just the state estimate part of the covariance matrix
	# pdeterminant = ( p[0][0] * ( (p[1][1]*p[2][2]) - (p[2][1]*p[1][2]) )) - ( p[0][1] * ( (p[1][0]*p[2][2]) - (p[2][0]*p[1][2]) )) + ( p[0][2] * ( (p[1][0]*p[2][1]) - (p[1][1]*p[2][0]) ))
	# uncertainty = math.sqrt(abs(pdeterminant))


def publishPoseWithCovariance(event):
	pose = PoseWithCovarianceStamped()
	pose.x = round(state[0][0], 3)  #change here
	pose.y = round(state[1][0], 3)
	pose.theta = round(state[2][0], 4)
	pose.covar = [p[0][0], p[0][1], p[0][2], p[1][0], p[1][1], p[1][2], p[2][0], p[2][1], p[2][2]]
	posePublisher.publish(pose)


def ROSNode():
	rospy.init_node('LKM_EncodersCubeBot', anonymous=False) 
	rospy.Subscriber('/EncodersCubeBot/odometry', Point, LKMPredictionStep)
	# rospy.Subscriber('/EncodersCubeBot/beacon_data', PointStamped, LKMUpdateStepWithBeaconData)

	# maybe move this into the prediction or update step function
	timer = rospy.Timer(rospy.Duration(PUBLISHER_TIMER_DURATION), publishPoseWithCovariance)            # timer to publish pose estimate at 2Hz
	
	rospy.spin()	#  go into an infinite loop until it receives a shutdown signal (Ctrl+C)
	timer.shutdown()


if __name__ == '__main__':
	try:
		ROSNode()
	except rospy.ROSInterruptException:
		pass