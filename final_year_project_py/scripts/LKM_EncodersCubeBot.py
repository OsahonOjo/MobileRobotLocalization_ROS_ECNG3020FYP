#!/usr/bin/env python
"""
@file LKM_EncodersCubeBot.py
@brief This script: receives encoder data from CoppeliaSim, uses it for pose and covariance estimation (dead-reckoning using EKF), then publishes the current pose estimate and uncertainty twice per second
"""


from __future__ import print_function
import json
import rospy, math, numpy
from geometry_msgs.msg import Point, PointStamped, PoseWithCovarianceStamped
# from final_year_project_py.msg import PoseWithCovarianceAndUncertainty


WHEEL_RADIUS = 0.035   			# in meters
WHEELBASE = 0.1					# distance between wheels (meters)

# for encoder
PPR = 11                        # 11 pulses (22 transitions) per rotation
ANGLE_PER_TICK = (2*math.pi)/(PPR*2)	# 1 tick per transition
previousLeftWheelTicks = 0
previousRightWheelTicks = 0

# my guess for error constants: accuracy/max = 45mm/12000mm = 0.00375
KR = 0.00375                 		# error constant: right wheel       
KL = 0.00375                 		# error constant: left wheel

BEACON_DETECTOR_RESOLUTION = 5*(math.pi/180)	# radians
BEACON_DETECTOR_STD_DEVIATION = 0.34 * BEACON_DETECTOR_RESOLUTION
BEACON_DETECTOR_VARIANCE = BEACON_DETECTOR_STD_DEVIATION**2
beaconDataCovariance = numpy.array([[0, 0], [0, BEACON_DETECTOR_VARIANCE]])
bearingNoiseSamples = numpy.random.default_rng().normal(0, BEACON_DETECTOR_VARIANCE, 1000)
bearingNoiseSamplesIndex = 0

initialPose2D = {}
beaconPositions = {}
with open('initialValues.json') as json_file:
    data = json.load(json_file)
    initialPose2D = data['EncodersCubeBotInitialPose']
    beaconPositions = { "beacon1": data['Beacon1Position'], "beacon2": data['Beacon2Position'] }

state = numpy.array([[initialPose2D['x']], [initialPose2D['y']], [initialPose2D['theta']]])
p = numpy.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
uncertainty = 0

posePublisher = rospy.Publisher('/EncodersCubeBot/pose_estimate', PoseWithCovarianceStamped, queue_size=10)  


# NOTE: tightly-coupled function: modifies 5, uses several other global variables
def LKMPredictionStep(odometry):
	currentLeftWheelTicks = odometry['x']
	currentRightWheelTicks = odometry['y']
	global previousLeftWheelTicks, previousRightWheelTicks, state, p, uncertainty

	# 1. estimate state

	leftTicksDifference = currentLeftWheelTicks - previousLeftWheelTicks
	rightTicksDifference = currentRightWheelTicks - previousRightWheelTicks
	
	# TODO: how to sample random (Gaussian) noise for encoder: del_sl += error_sl ?
	delSl = WHEEL_RADIUS * (leftTicksDifference * ANGLE_PER_TICK)
	errorSl = KL*abs(delSl)
	delSr = WHEEL_RADIUS * (rightTicksDifference * ANGLE_PER_TICK)
	errorSr = KR*abs(delSr)
	
	delS = (delSr + delSl) / 2
	delTheta = (delSr - delSl) / WHEELBASE
	theta = state[2][0]
	thetaArg = theta + (delTheta/2)	

	state = state + [[delS*math.cos(thetaArg)], [delS*math.sin(thetaArg)], [delTheta]]
	odometryCovariance = numpy.array([[errorSr, 0], [0, errorSl]])	


	# 2. predict covariance

	# partial derivative of configuration w.r.t state (x,y,z)
	fp = numpy.array([[1, 0, -1*delS*math.sin(thetaArg)], [0, 1, delS*math.cos(thetaArg)], [0, 0, 1]])
	fpT = fp.transpose()

	# partial derivative of configuration w.r.t odometry (del_sl, del_sr)
	fs = numpy.array([[(0.5*math.cos(thetaArg))-(delS/(2*WHEELBASE))*math.sin(thetaArg), (0.5*math.cos(thetaArg))+((delS/(2*WHEELBASE))*math.sin(thetaArg))], [(0.5*math.sin(thetaArg))+(delS/(2*WHEELBASE))*math.cos(thetaArg), (0.5*math.sin(thetaArg))-((delS/(2*WHEELBASE))*math.cos(thetaArg))], [(1/WHEELBASE), (-1/WHEELBASE)]])
	fsT = fs.transpose()
	
	p = fp.dot(p).dot(fpT) + fs.dot(odometryCovariance).dot(fsT)

	# determinant of just the state estimate part of the covariance matrix
	robotPoseDeterminant = ( p[0][0] * ( (p[1][1]*p[2][2]) - (p[2][1]*p[1][2]) )) - ( p[0][1] * ( (p[1][0]*p[2][2]) - (p[2][0]*p[1][2]) )) + ( p[0][2] * ( (p[1][0]*p[2][1]) - (p[1][1]*p[2][0]) ))
	uncertainty = math.sqrt(abs(robotPoseDeterminant))

	previousLeftWheelTicks = currentLeftWheelTicks
	previousRightWheelTicks = currentRightWheelTicks

	print("state: \n", state)
	print("Covariance matrix: \n", p)
	print("uncertainty: \n", uncertainty)

	# publishPoseWithCovariance(state, p)


# beaconData: { header={ stamp={ secs, nsecs }, frame_id }, point={ x (bearing [degrees]), y, z } })
def LKMUpdateStepWithBeaconData(beaconData):
	global state, p, uncertainty, bearingNoiseSamplesIndex

	# 3. calculate innovation: measured angle - expected angle + sensor noise

	measuredBearing = beaconData['point']['x']
	beaconX = beaconPositions[beaconData['header']['frame_id']]['x']
	beaconY = beaconPositions[beaconData['header']['frame_id']]['y']
	robotX = state[0][0]
	robotY = state[1][0]
	robotTheta = state[2][0]
	expectedBeaconRange = math.sqrt((beaconX - robotX)**2 + (beaconY - robotY)**2)

	measuredBeaconData = numpy.array([[expectedBeaconRange], [measuredBearing]])	# radians
	expectedBeaconData = numpy.array([[expectedBeaconRange], [math.atan2((beaconY - robotY), (beaconX - robotX)) - robotTheta]])
	
	beaconDataNoise = numpy.array([[0], [bearingNoiseSamples[bearingNoiseSamplesIndex]]])
	bearingNoiseSamplesIndex = 0 if bearingNoiseSamplesIndex == len(bearingNoiseSamples)-1 else bearingNoiseSamplesIndex + 1
	innovation = measuredBeaconData - expectedBeaconData + beaconDataNoise


	# 4. update state estimate

	hw = numpy.array([[1, 0], [0, 1]])
	hwT = hw.transpose()
	r = expectedBeaconRange
	hx = numpy.array([[-(beaconX - robotX)/r, -(beaconY - robotY)/r, 0], [(beaconY - robotY)/r**2, -(beaconX - robotX)/r**2, -1]])
	hxT = hx.transpose()
	print("hw: \n", hw, "\nhx: \n", hx)

	# TODO
	s = hx*p*hxT + hw*beaconDataCovariance*hwT

	sInverse = numpy.linalg.inv(s)	# A non-square matrix does not have an inverse
	k = p*hxT*sInverse
	state = state + (k*innovation)


	# 5. update covariance

	p = p - k*hx*p
	robotPoseDeterminant = ( p[0][0] * ( (p[1][1]*p[2][2]) - (p[2][1]*p[1][2]) )) - ( p[0][1] * ( (p[1][0]*p[2][2]) - (p[2][0]*p[1][2]) )) + ( p[0][2] * ( (p[1][0]*p[2][1]) - (p[1][1]*p[2][0]) ))
	uncertainty = math.sqrt(abs(robotPoseDeterminant))

	print("state: \n", state)
	print("Covariance matrix: \n", p)
	print("uncertainty: \n", uncertainty)

	# publishPoseWithCovariance(state, p)


# https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
def eulerToQuaternion(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    return x, y, z, w


def publishPoseWithCovariance(pose2D, covariance):
	pose = {}
	pose.position = { "x": round(pose2D[0][0], 3), "y": round(pose2D[1][0], 3), "z": 0 }
	qx, qy, qz, qw = eulerToQuaternion(0, 0, round(pose2D[2][0]))
	pose.orientation = { "x": qx, "y": qy, "z": qz, "w": qw }
	
	poseStamped = PoseWithCovarianceStamped()
	poseStamped.pose.pose = pose
	poseStamped.pose.covariance = covariance.tolist()
	posePublisher.publish(poseStamped)


def ROSNode():
	rospy.init_node('LKM_EncodersCubeBot', anonymous=False) 
	rospy.Subscriber('/EncodersCubeBot/odometry', Point, LKMPredictionStep)
	rospy.Subscriber('/EncodersCubeBot/beacon_data', PointStamped, LKMUpdateStepWithBeaconData)
	rospy.spin()	#  go into an infinite loop until it receives a shutdown signal (Ctrl+C)


if __name__ == '__main__':
	try:
		# ROSNode()
		LKMPredictionStep({"x": 1, "y": 2, "z": -1})
		LKMUpdateStepWithBeaconData({ "header": { "frame_id": "beacon1" }, "point": { "x": math.pi/5 } })
	except rospy.ROSInterruptException:
		pass