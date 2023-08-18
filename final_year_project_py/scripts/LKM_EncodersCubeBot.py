#!/usr/bin/env python
"""
@file LKM_EncodersCubeBot.py
@brief This script receives encoder data from CoppeliaSim, uses it for pose and covariance estimation (localization using EKF), then publishes the current pose estimate and uncertainty twice per second
"""


from __future__ import print_function
import json, rospy, math, numpy
from geometry_msgs.msg import Point, PointStamped, PoseWithCovarianceStamped
# from final_year_project_py.msg import PoseWithCovarianceAndUncertainty


WHEEL_RADIUS = 0.035   			# in meters
WHEELBASE = 0.1					# distance between wheels (meters)
PPR = 11                        # encoders: 11 pulses (22 transitions) per rotation
ANGLE_PER_TICK = (2*math.pi)/(PPR*2)	# encoders: 1 tick per transition
KR = 0.00375                 		# right wheel error constant guess: accuracy/max = 45mm/12000mm = 0.00375
KL = 0.00375                 		# left wheel error constant guess: accuracy/max = 45mm/12000mm = 0.00375
BEACON_DETECTOR_RESOLUTION = 5*(math.pi/180)	# radians
BEACON_DETECTOR_STD_DEVIATION = 0.34 * BEACON_DETECTOR_RESOLUTION
BEACON_DETECTOR_VARIANCE = BEACON_DETECTOR_STD_DEVIATION**2

prevTotalLeftTicks = 0
prevTotalRightTicks = 0
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


def ROSNode():
	rospy.init_node('LKM_EncodersCubeBot', anonymous=False) 
	rospy.Subscriber('/EncodersCubeBot/odometry', Point, lambda odometry : predictionStepCallback(odometry.x, odometry.y))
	rospy.Subscriber('/EncodersCubeBot/beacon_data', PointStamped, LKMUpdateStepWithBeaconData)
	rospy.spin()	#  go into an infinite loop until it receives a shutdown signal (Ctrl+C)


# NOTE: coupled to several global variables (see global keyword below)
def predictionStepCallback(totalLeftTicks, totalRightTicks):
	global state, p, uncertainty
	
	newLeftTicks, newRightTicks = getNewEncoderTicksCount(totalLeftTicks, getPrevTotalLeftTicks(), totalRightTicks, getPrevTotalRightTicks())
	delSl, delSr = encoderTicksToAngularDistances(newLeftTicks, newRightTicks)
	delS, delTheta = calculateControlInput(delSl, delSr)
	state, thetaArg = motionModel(state, delS, delTheta)
	
	errorSl, errorSr = calculateControlError(delSl, delSr)
	odometryCovariance = numpy.diag([errorSr, errorSl]).reshape((2,2))
	fp, fpT, fs, fsT = calculateJacobians(delS, thetaArg)
	p = (fp @ p @ fpT) + (fs @ odometryCovariance @ fsT)
	uncertainty = pose2DUncertainty(p)

	setprevTotalLeftTicks(totalLeftTicks)
	setPrevTotalRightTicks(totalRightTicks)
	return True

# NOTE: following four functions are coupled to global variables (see global keyword below)
def getPrevTotalLeftTicks(): 
	return prevTotalLeftTicks

def setprevTotalLeftTicks(ticks):
	global prevTotalLeftTicks
	prevTotalLeftTicks = ticks

def getPrevTotalRightTicks():
	return prevTotalRightTicks

def setPrevTotalRightTicks(ticks):
	global prevTotalRightTicks
	prevTotalRightTicks = ticks

def getNewEncoderTicksCount(currentTotalLeftTicks, prevTotalLeftTicks, currentTotalRightTicks, prevTotalRightTicks):
	newLeftTicks = currentTotalLeftTicks - prevTotalLeftTicks
	newRightTicks = currentTotalRightTicks - prevTotalRightTicks
	return newLeftTicks, newRightTicks	

def encoderTicksToAngularDistances(newLeftWheelTicks, newRightWheelTicks, wheelRadius=WHEEL_RADIUS, anglePerTick=ANGLE_PER_TICK):
	"""
    Calculates the angular distance travelled by the differential robot's wheels using new encoder ticks ("new encoder ticks" referring to only the number of ticks counted in the last interval, not to a running total)
    :param newLeftWheelTicks: 	number of new encoder ticks on the left wheel
	:param newRightWheelTicks: 	number of new encoder ticks on the right wheel
	:param wheelRadius: 		radius of the wheels the encoders are attached to
	:param anglePerTick: 		angle covered per encoder tick
    :returns: 					angular distances delSl, delSr
    """
	delSl = wheelRadius * (newLeftWheelTicks * anglePerTick)
	delSr = wheelRadius * (newRightWheelTicks * anglePerTick)
	return delSl, delSr

def calculateControlInput(delSl, delSr, wheelbase=WHEELBASE):
	"""
    Computes the control function inputs using the angular distances travelled by the left and right wheels of the differential robot
    :param delSl: 		tiny (hence del) distance travelled by the left wheel
    :param delSr: 		tiny (hence del) distance travelled by the right wheel
    :param wheelbase: 	distance between differential robot's two wheels
    :returns: 			control inputs [delS; delTheta]
    """
	delS = (delSr + delSl) / 2
	delTheta = (delSr - delSl) / wheelbase
	return delS, delTheta

def calculateControlError(delSl, delSr, kl=KL, kr=KR):
	"""
    Computes the errors for the control function inputs using error constants, and the angular distances travelled by the left and right wheels of the differential robot
    :param delSl: 	tiny (hence del) distance travelled by the left wheel
    :param delSr: 	tiny (hence del) distance travelled by the right wheel
    :param kl: 		error constant for left wheel
    :param kr: 		error constant for right wheel
    :returns: 		error values errorSl, errorSr
    """
	errorSl = kl*abs(delSl)
	errorSr = kr*abs(delSr)
	return errorSl, errorSr

def motionModel(state, delS, delTheta):
	"""
	Computes the motion model based on current state and input function
	:param state: 		3x1 pose estimation [x; y; z]
	:param delS: 		control function inputs: tiny (hence del) distance travelled by imaginary point along robot's centerline
	:param delTheta: 	control function inputs: tiny (hence del) change in robot's bearing/orientation
	:returns: 			the resulting state after the control function is applied
	:returns:			a value thetaArg needed for other calculation
	"""
	theta = state[2][0]
	thetaArg = theta + (delTheta/2)
	state = state + [[delS*math.cos(thetaArg)], [delS*math.sin(thetaArg)], [delTheta]]
	return state, thetaArg

def calculateJacobians(delS, thetaArg, wheelbase=WHEELBASE):
	"""
    Calculates the angular distance travelled by the differential robot's wheels using new encoder ticks ("new encoder ticks" referring to only the number of ticks counted in the last interval, not to a running total)
    :param delS: 		control function inputs: tiny (hence del) distance travelled by imaginary point along robot's centerline
	:param thetaArg: 	a value equal to theta + (delTheta/2), needed for other calculation
	:param wheelbase: 	distance between differential robot's two wheels
    :returns: 			jacobians: partial derivative of configuration w.r.t state (x,y,z) (fp) and odometry (delSl, delSr) (fs)
    """
	fp = numpy.array([[1, 0, -1*delS*math.sin(thetaArg)], [0, 1, delS*math.cos(thetaArg)], [0, 0, 1]]).reshape((3,3))
	fpT = fp.transpose().reshape((3,3))
	fs = numpy.array([[(0.5*math.cos(thetaArg))-(delS/(2*wheelbase))*math.sin(thetaArg), (0.5*math.cos(thetaArg))+((delS/(2*wheelbase))*math.sin(thetaArg))], [(0.5*math.sin(thetaArg))+(delS/(2*wheelbase))*math.cos(thetaArg), (0.5*math.sin(thetaArg))-((delS/(2*wheelbase))*math.cos(thetaArg))], [(1/wheelbase), (-1/wheelbase)]]).reshape((3,2))
	fsT = fs.transpose().reshape((2,3))	
	return fp, fpT, fs, fsT

def pose2DUncertainty(p):
	"""
	Calculates the overall uncertainty in the state estimate
	:param p: 		3x3 the predicted covariance
	:returns:     	the overall uncertainty in the robot pose estimate
	"""
	if numpy.shape(p) != (3,3):
		return None
	determinant = ( p[0][0] * ( (p[1][1]*p[2][2]) - (p[2][1]*p[1][2]) )) - ( p[0][1] * ( (p[1][0]*p[2][2]) - (p[2][0]*p[1][2]) )) + ( p[0][2] * ( (p[1][0]*p[2][1]) - (p[1][1]*p[2][0]) ))
	return math.sqrt(abs(determinant))


# def update(xEst, PEst, u, z, initP):


# NOTE: tightly-coupled function: modifies 5, uses several other global variables
def LKMPredictionStep(odometry):
	# currentLeftWheelTicks = odometry['x']
	currentLeftWheelTicks = odometry.x
	# currentRightWheelTicks = odometry['y']
	currentRightWheelTicks = odometry.y
	global prevTotalLeftTicks, prevTotalRightTicks, state, p, uncertainty

	# 1. estimate state

	leftTicksDifference = currentLeftWheelTicks - prevTotalLeftTicks
	rightTicksDifference = currentRightWheelTicks - prevTotalRightTicks
	
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

	prevTotalLeftTicks = currentLeftWheelTicks
	prevTotalRightTicks = currentRightWheelTicks

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
	# -> when multiplying with 1x1 scalar, use * (element-wise mult.) instead of @ (matrix mult.)
	#    you'll still get the result you expect
	# -> ALWAYS use reshape() to explicitly set matrix dimensions

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

	print("From Update Step")
	pose2D = [state[0][0], state[1][0], state[2][0]]
	covariance1x36 = numpy.append(numpy.reshape(p, (1,9)), numpy.zeros(27))
	print("pose2D: \n", pose2D)
	print("covariance1x36: \n", covariance1x36)
	publishPoseWithCovariance(pose2D, covariance1x36)


def publishPoseWithCovariance(pose2D, covariance1D):
	# LESSON: 
	# -> initialize a message type and then set each property using its full dot notation name
	#    doing this will prevent you from getting a bunch of "AttributeError: 'dict' object has no attribute 'attribute'" errors

	poseStamped = PoseWithCovarianceStamped()
	qx, qy, qz, qw = eulerToQuaternion(0, 0, round(pose2D[2]))	
	print("poseStamped init: \n", poseStamped)
	poseStamped.pose.pose.position.x = round(pose2D[0], 3)
	poseStamped.pose.pose.position.y = round(pose2D[1], 3)
	poseStamped.pose.pose.position.z = 0
	poseStamped.pose.pose.orientation.x = qx
	poseStamped.pose.pose.orientation.y = qy
	poseStamped.pose.pose.orientation.z = qz
	poseStamped.pose.pose.orientation.w = qw
	poseStamped.pose.covariance = covariance1D.tolist()
	print("poseStamped: \n", poseStamped)
	posePublisher.publish(poseStamped)

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


if __name__ == '__main__':
	try:
		ROSNode()
		# LKMPredictionStep({"x": 1, "y": 2, "z": -1})
		# LKMUpdateStepWithBeaconData({ "header": { "frame_id": "beacon1" }, "point": { "x": math.pi/5 } })
	except rospy.ROSInterruptException:
		pass