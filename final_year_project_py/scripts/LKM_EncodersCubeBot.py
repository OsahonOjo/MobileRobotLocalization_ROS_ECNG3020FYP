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
beaconDataCovariance = numpy.array([BEACON_DETECTOR_VARIANCE]).reshape((1,1))
bearingNoiseSamples = numpy.random.default_rng().normal(0, BEACON_DETECTOR_VARIANCE, 1000)
bearingNoiseSamplesIndex = 0

initialPose2D = {}
beaconPositions = {}
with open('/home/osahon/catkin_ws/src/final_year_project_py/scripts/initialValues.json') as json_file:
    data = json.load(json_file)
    initialPose2D = data['EncodersCubeBotInitialPose']
    beaconPositions = { "beacon1": data['Beacon1Position'], "beacon2": data['Beacon2Position'] }

state = numpy.array([[initialPose2D['x']], [initialPose2D['y']], [initialPose2D['theta']]]).reshape((3,1))
p = numpy.diag([0, 0, 0]).reshape((3,3))
uncertainty = 0

posePublisher = rospy.Publisher('/EncodersCubeBot/pose_estimate', PoseWithCovarianceStamped, queue_size=10)  


# NOTE: tightly-coupled function: modifies 5, uses several other global variables
def LKMPredictionStep(odometry):
	# currentLeftWheelTicks = odometry['x']
	currentLeftWheelTicks = odometry.x
	# currentRightWheelTicks = odometry['y']
	currentRightWheelTicks = odometry.y
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
	odometryCovariance = numpy.diag([errorSr, errorSl]).reshape((2,2))


	# 2. predict covariance

	# partial derivative of configuration w.r.t state (x,y,z)
	fp = numpy.array([[1, 0, -1*delS*math.sin(thetaArg)], [0, 1, delS*math.cos(thetaArg)], [0, 0, 1]]).reshape((3,3))
	fpT = fp.transpose().reshape((3,3))
	# print("fpT: ", numpy.shape(fpT))  # fpT:  (3, 3)

	# partial derivative of configuration w.r.t odometry (del_sl, del_sr)
	fs = numpy.array([[(0.5*math.cos(thetaArg))-(delS/(2*WHEELBASE))*math.sin(thetaArg), (0.5*math.cos(thetaArg))+((delS/(2*WHEELBASE))*math.sin(thetaArg))], [(0.5*math.sin(thetaArg))+(delS/(2*WHEELBASE))*math.cos(thetaArg), (0.5*math.sin(thetaArg))-((delS/(2*WHEELBASE))*math.cos(thetaArg))], [(1/WHEELBASE), (-1/WHEELBASE)]]).reshape((3,2))
	fsT = fs.transpose().reshape((2,3))
	# print("fsT: ", numpy.shape(fsT))  # fsT:  (2, 3)
	
	p = (fp @ p @ fpT) + (fs @ odometryCovariance @ fsT)

	# overall uncertainty for pose (from Corke, pg. 160)
	robotPoseDeterminant = ( p[0][0] * ( (p[1][1]*p[2][2]) - (p[2][1]*p[1][2]) )) - ( p[0][1] * ( (p[1][0]*p[2][2]) - (p[2][0]*p[1][2]) )) + ( p[0][2] * ( (p[1][0]*p[2][1]) - (p[1][1]*p[2][0]) ))
	uncertainty = math.sqrt(abs(robotPoseDeterminant))

	previousLeftWheelTicks = currentLeftWheelTicks
	previousRightWheelTicks = currentRightWheelTicks

	# print("state: \n", state)
	# print("Covariance matrix: \n", p)
	# print("uncertainty: \n", uncertainty)

	print("From Prediction Step")
	publishPoseWithCovariance(state, p)


# beaconData: { header={ stamp={ secs, nsecs }, frame_id }, point={ x (bearing [degrees]), y, z } })
def LKMUpdateStepWithBeaconData(beaconData):
	global state, p, uncertainty, bearingNoiseSamplesIndex

	# 3. calculate innovation: measured angle - expected angle + sensor noise

	# measuredBearing = beaconData['point']['x']
	measuredBearing = beaconData.point.x	
	# beaconX = beaconPositions[beaconData['header']['frame_id']]['x']
	beaconX = beaconPositions[beaconData.header.frame_id]['x']
	# beaconY = beaconPositions[beaconData['header']['frame_id']]['y']
	beaconY = beaconPositions[beaconData.header.frame_id]['y']
	robotX = state[0][0]
	robotY = state[1][0]
	robotTheta = state[2][0]
	expectedBeaconRange = math.sqrt((beaconX - robotX)**2 + (beaconY - robotY)**2)

	measuredBeaconData = numpy.array([measuredBearing]).reshape((1,1))	# radians
	expectedBeaconData = numpy.array([math.atan2((beaconY - robotY), (beaconX - robotX)) - robotTheta]).reshape((1,1))
	
	beaconDataNoise = numpy.array([bearingNoiseSamples[bearingNoiseSamplesIndex]]).reshape((1,1))
	bearingNoiseSamplesIndex = 0 if bearingNoiseSamplesIndex == len(bearingNoiseSamples)-1 else bearingNoiseSamplesIndex + 1
	innovation = measuredBeaconData - expectedBeaconData + beaconDataNoise


	# 4. update state estimate

	# hw = numpy.array([[1, 0], [0, 1]])
	print("p: \n", p)
	print("beaconDataCovariance: \n", beaconDataCovariance)

	r = expectedBeaconRange
	print("r: \n", r)

	hw = numpy.array([1]).reshape((1,1))  # hw = dh_bearing/dw_bearing => 1x1 scalar
	print("hw: \n", hw)
	hwT = numpy.transpose(hw).reshape((1,1))
	print("hwT: ", numpy.shape(hwT))

	# hx = numpy.array([[-(beaconX - robotX)/r, -(beaconY - robotY)/r, 0], [(beaconY - robotY)/r**2, -(beaconX - robotX)/r**2, -1]])
	hx = numpy.array([(beaconY - robotY)/r**2, -(beaconX - robotX)/r**2, -1]).reshape((1,3))  # 1x3 vector
	print("hx: \n", hx)
	hxT = numpy.transpose(hx).reshape((3,1))
	print("hxT: ", numpy.shape(hxT))
	
	# LESSON: 
	# when multiplying with 1x1 scalar, use * (element-wise mult.) instead of @ (matrix mult.)
	# you'll still get the result you expect
	# ALWAYS use reshape() to explicitly set matrix dimensions

	# s1 = hx @ p @ hxT  # 1x1 scalar
	# print("s1: \n", s1)
	# s2 = hw * beaconDataCovariance * hwT  # changed @ to *
	# 1x1 scalar
	# operator * which was used either for element-wise multiplication or matrix multiplication depending on the convention employed in that particular library/code. As a result, in the future, the operator * is meant to be used for element-wise multiplication only
	# print("s2: \n", s2)
	s = (hx @ p @ hxT) + (hw * beaconDataCovariance * hwT)  # 1x1 scalar
	print("s: \n", s)

	# s = hx @ p @ hxT + hw @ beaconDataCovariance @ hwT  # 1x1 scalar
	# print("s: \n", s)
	# inverse of a 1x1 matrix is simply the reciprical of the single entry in the matrix
	# try:
	# 	sInverse = numpy.linalg.inv(s)	# A non-square matrix does not have an inverse
	# except numpy.linalg.LinAlgError("Singular matrix") as e:
	# 	print(e)	# A square matrix is singular if and only if its determinant is 0
	sInverse = 1/s  # 1x1 scalar
	print("sInverse: \n", sInverse)
	
	# kscalar = hxT * sInverse
	# kscalar = numpy.reshape(kscalar, (3,1))
	# print("kscalar: ", numpy.shape(kscalar))
	# print(kscalar)

	k = p @ (hxT * sInverse)  # 3x3 3x1 1x1 => 3x3 3x1 => 3x1
	k = numpy.reshape(k, (3,1))
	print("k: ", numpy.shape(k))
	print(k)
	
	state = state + (k * innovation)  # 3x1 + (3x1 1x1) => 3x1
	print("state: \n", state)

	# 5. update covariance
	# psub = k @ (hx @ p)
	# psub = numpy.reshape(psub, (3,3))
	# print("psub: \n", psub)

	p = p - (k @ hx @ p)
	print("p: \n", p)

	robotPoseDeterminant = ( p[0][0] * ( (p[1][1]*p[2][2]) - (p[2][1]*p[1][2]) )) - ( p[0][1] * ( (p[1][0]*p[2][2]) - (p[2][0]*p[1][2]) )) + ( p[0][2] * ( (p[1][0]*p[2][1]) - (p[1][1]*p[2][0]) ))
	uncertainty = math.sqrt(abs(robotPoseDeterminant))

	# From Prediction Step
	# p: 
	# [[1.16729523e-06 3.13611861e-06 6.68494671e-05]
	# [3.13611861e-06 9.40076721e-06 1.98167016e-04]
	# [6.68494671e-05 1.98167016e-04 4.18230873e-03]]
	# beaconDataCovariance: 
	# [[0.00088034]]
	# r: 
	# 1.033969272961315
	# hw: 
	# [[1]]
	# hwT:  (1, 1)
	# hx: 
	# [[ 0.20245612 -0.94571894 -1.        ]]
	# hxT:  (3, 1)
	# s: 
	# [[0.00541766]]
	# sInverse: 
	# [[184.58152407]]
	# k:  (3, 1)
	# [[-0.012843  ]
	# [-0.03810179]
	# [-0.80407126]]
	# state: 
	# [[0.22252566]
	# [0.40676142]
	# [0.41427224]]
	# p: 
	# [[2.73691566e-07 4.85032883e-07 1.09029708e-05]
	# [4.85032883e-07 1.53569701e-06 3.21885582e-05]
	# [1.09029708e-05 3.21885582e-05 6.79625626e-04]]
	# From Update Step
	# p: 
	# [[2.73691566e-07 4.85032883e-07 1.09029708e-05]
	# [4.85032883e-07 1.53569701e-06 3.21885582e-05]
	# [1.09029708e-05 3.21885582e-05 6.79625626e-04]]
	# beaconDataCovariance: 
	# [[0.00088034]]
	# r: 
	# 0.3205065356189586
	# hw: 
	# [[1]]
	# hwT:  (1, 1)
	# hx: 
	# [[ 2.4944312   1.87419176 -1.        ]]
	# hxT:  (3, 1)
	# s: 
	# [[0.00139655]]
	# sInverse: 
	# [[716.04830459]]
	# k:  (3, 1)
	# [[-0.00666728]
	# [-0.02012131]
	# [-0.42397319]]
	# state: 
	# [[0.23454169]
	# [0.4430248 ]
	# [1.17837265]]
	# p: 
	# [[2.11611007e-07 2.97678952e-07 6.95526253e-06]
	# [2.97678952e-07 9.70278530e-07 2.02747053e-05]
	# [6.95526253e-06 2.02747053e-05 4.28590511e-04]]
	# From Update Step



	# [ERROR] [1690955382.326007]: bad callback: <function LKMUpdateStepWithBeaconData at 0x7fae8f606160>
	# Traceback (most recent call last):
	# File "/opt/ros/noetic/lib/python3/dist-packages/rospy/topics.py", line 750, in _invoke_callback
	# 	cb(msg)
	# File "/home/osahon/catkin_ws/src/final_year_project_py/scripts/LKM_EncodersCubeBot.py", line 150, in LKMUpdateStepWithBeaconData
	# 	s2 = hw @ beaconDataCovariance @ hwT
	# ValueError: matmul: Input operand 0 does not have enough dimensions (has 0, gufunc core with signature (n?,k),(k,m?)->(n?,m?) requires 1)

	print("From Update Step")
	publishPoseWithCovariance(state, p)


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
	pose = {"position": {}, "orientation": {}}
	pose['position'] = { "x": round(pose2D[0][0], 3), "y": round(pose2D[1][0], 3), "z": 0 }
	qx, qy, qz, qw = eulerToQuaternion(0, 0, round(pose2D[2][0]))
	pose['orientation'] = { "x": qx, "y": qy, "z": qz, "w": qw }
	
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
		ROSNode()
		# LKMPredictionStep({"x": 1, "y": 2, "z": -1})
		# LKMUpdateStepWithBeaconData({ "header": { "frame_id": "beacon1" }, "point": { "x": math.pi/5 } })
	except rospy.ROSInterruptException:
		pass