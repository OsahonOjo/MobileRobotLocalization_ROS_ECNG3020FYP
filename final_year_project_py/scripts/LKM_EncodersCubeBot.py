#!/usr/bin/env python
"""
@file LKM_EncodersCubeBot.py
@brief This script receives encoder and beacon data from CoppeliaSim, uses it for pose and covariance estimation (localization using EKF), then publishes the current pose estimate and uncertainty twice per second
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

gPrevTotalLeftTicks = 0
gPrevTotalRightTicks = 0
gBeaconDataCovariance = numpy.array([BEACON_DETECTOR_VARIANCE]).reshape((1,1))
gBearingNoiseSamples = numpy.random.default_rng().normal(0, BEACON_DETECTOR_VARIANCE, 1000)
gBearingNoiseSamplesIndex = 0
gInitialPose2D = {}
gBeaconPositions = {}
with open('/home/osahon/catkin_ws/src/final_year_project_py/scripts/initialValues.json') as json_file:
    data = json.load(json_file)
    gInitialPose2D = data['EncodersCubeBotInitialPose']
    gBeaconPositions = { "beacon1": data['Beacon1Position'], "beacon2": data['Beacon2Position'] }
gState = numpy.array([[gInitialPose2D['x']], [gInitialPose2D['y']], [gInitialPose2D['theta']]]).reshape((3,1))
gP = numpy.diag([0, 0, 0]).reshape((3,3))
gUncertainty = 0
gPosePublisher = rospy.Publisher('/EncodersCubeBot/pose_estimate', PoseWithCovarianceStamped, queue_size=10)
gEKFPredictionStepPosePublisher = rospy.Publisher('/EncodersCubeBot/EKF/prediction_step', PoseWithCovarianceStamped, queue_size=10)  
gEKFUpdateStepPosePublisher = rospy.Publisher('/EncodersCubeBot/EKF/update_step', PoseWithCovarianceStamped, queue_size=10)  


# Pass-by-object-reference: in Python, “Object references are passed by value”

def ROSNode():
	rospy.init_node('LKM_EncodersCubeBot', anonymous=False) 
	rospy.Subscriber('/EncodersCubeBot/odometry', Point, lambda odometry : predictionStepCallback(odometry.x, odometry.y))
	# rospy.Subscriber('/EncodersCubeBot/beacon_data', PointStamped, lambda beaconData : updateStepCallback(beaconData))
	rospy.spin()	#  go into an infinite loop until it receives a shutdown signal (Ctrl+C)

# NOTE: modifies global variables (state, p, uncertainty, prevTotalLeftTicks, prevTotalRightTicks)
# def predictionStepCallback(state, p, totalLeftTicks, prevTotalLeftTicks, totalRightTicks, prevTotalRightTicks):
def predictionStepCallback(totalLeftTicks, totalRightTicks):
	"""
    Callback function for messages received on /EncodersCubeBot/odometry topic.
	Performs the prediction step of EKF Localization (see predictionStep() function); publishes the state estimate and its covariance; and performs several side effects (see comment-wrapped block below)
    """
	print("totalLeftTicks, totalRightTicks: ", totalLeftTicks, totalRightTicks)
	if totalLeftTicks == 0 and totalRightTicks == 0: return
	state, p = predictionStep(gState, gP, totalLeftTicks, gPrevTotalLeftTicks, totalRightTicks, gPrevTotalRightTicks)
	## [SIDE EFFECTS]
	gSetState(state)
	gSetStateCovariance(p)
	gSetUncertainty(p)
	gSetprevTotalLeftTicks(totalLeftTicks)
	gSetPrevTotalRightTicks(totalRightTicks)
	pose2D = [state[0][0], state[1][0], state[2][0]]
	covariance1x36 = numpy.append(numpy.reshape(p, (1,9)), numpy.zeros(27))
	publishPoseWithCovariance(gPosePublisher, pose2D, covariance1x36)
	publishPoseWithCovariance(gEKFPredictionStepPosePublisher, pose2D, covariance1x36)
	## [SIDE EFFECTS]
	print("[predict] state: \n", gState)
	print("[predict] p: \n", gP)
	print("[predict] uncertainty: ", gUncertainty)

def predictionStep(state, p, totalLeftTicks, prevTotalLeftTicks, totalRightTicks, prevTotalRightTicks):
	"""
    Performs the prediction step of EKF Localization: estimates the state and its covariance (global variables: state, p)
    :param state: 				current state estimate
    :param p: 					covariance matrix for current state estimate
    :param totalLeftTicks: 		running total, total number of encoder ticks on the left wheel
    :param totalRightTicks: 	running total, total number of encoder ticks on the right wheel
    :param prevTotalLeftTicks: 	previous total number of encoder ticks on the left wheel
    :param prevTotalRightTicks: previous total number of encoder ticks on the right wheel
    :returns:					state estimate and covariance
    """	
	newLeftTicks, newRightTicks = getNewEncoderTicksCount(totalLeftTicks, prevTotalLeftTicks, totalRightTicks, prevTotalRightTicks)
	delSl, delSr = encoderTicksToDistance(newLeftTicks, newRightTicks)
	delS, delTheta = calculateControlInput(delSl, delSr)
	theta = state[2][0]
	thetaArg = theta + (delTheta/2)
	state = motionModel(state, delS, delTheta)
	errorSl, errorSr = calculateControlError(delSl, delSr)
	odometryCovariance = numpy.diag([errorSr, errorSl]).reshape((2,2))
	fp, fpT, fs, fsT = calculatePredictionStepJacobians(delS, thetaArg)
	p = (fp @ p @ fpT) + (fs @ odometryCovariance @ fsT)
	return state, p
	

# TODO: MOVE THESE TO NEAR END OF FILE
## [GLOBAL VARIABLE MANIPULATORS]: get/set, otherwise modify several global variables
def gSetState(state):
	global gState
	gState = state

def gSetStateCovariance(p):
	global gP
	gP = p

def gSetprevTotalLeftTicks(ticks):
	global gPrevTotalLeftTicks
	gPrevTotalLeftTicks = ticks

def gSetPrevTotalRightTicks(ticks):
	global gPrevTotalRightTicks
	gPrevTotalRightTicks = ticks

def gSetUncertainty(covarianceMatrix3x3):
	global gUncertainty
	gUncertainty = pose2DUncertainty(covarianceMatrix3x3)

def gIncrementBearingNoiseSampleIndex():
	global gBearingNoiseSamplesIndex
	if gBearingNoiseSamplesIndex == len(gBearingNoiseSamples)-1:
		return 0
	return gBearingNoiseSamplesIndex + 1

def gGetBearingNoiseSample():
	return gBearingNoiseSamples[gBearingNoiseSamplesIndex]
## [GLOBAL VARIABLE MANIPULATORS]

def ungroupStateNumpyArray(state):
	return state[0][0], state[1][0], state[2][0]

def getNewEncoderTicksCount(currentTotalLeftTicks, prevTotalLeftTicks, currentTotalRightTicks, prevTotalRightTicks):
	"""
	Calculates the difference between the previous and current total ticks on the left and right wheels of the differential-drive robot
	"""
	newLeftTicks = currentTotalLeftTicks - prevTotalLeftTicks
	newRightTicks = currentTotalRightTicks - prevTotalRightTicks
	return newLeftTicks, newRightTicks	

def encoderTicksToDistance(newLeftWheelTicks, newRightWheelTicks, wheelRadius=WHEEL_RADIUS, anglePerTick=ANGLE_PER_TICK):
	"""
    Calculates the angular distance travelled by the differential-drive robot's wheels using new encoder ticks ("new encoder ticks" referring to only the number of ticks counted in the last interval, not to a running total)
    :param newLeftWheelTicks: 	number of new encoder ticks on the left wheel
	:param newRightWheelTicks: 	number of new encoder ticks on the right wheel
	:param wheelRadius: 		radius of the wheels the encoders are attached to
	:param anglePerTick: 		angle covered per encoder tick
    :returns: 					distances delSl, delSr
    """
	delSl = wheelRadius * (newLeftWheelTicks * anglePerTick)
	delSr = wheelRadius * (newRightWheelTicks * anglePerTick)
	return delSl, delSr

def calculateControlInput(delSl, delSr, wheelbase=WHEELBASE):
	"""
    Computes the control function inputs using the distances travelled by the left and right wheels of the differential-drive robot
    :param delSl: 		tiny (hence del) distance travelled by the left wheel
    :param delSr: 		tiny (hence del) distance travelled by the right wheel
    :param wheelbase: 	distance between differential-drive robot's two wheels
    :returns: 			control inputs [delS; delTheta]
    """
	delS = (delSr + delSl) / 2
	delTheta = (delSr - delSl) / wheelbase
	return delS, delTheta

def calculateControlError(delSl, delSr, kl=KL, kr=KR):
	"""
    Computes the errors for the control function inputs using error constants, and the distances travelled by the left and right wheels of the differential-drive robot
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
	"""
	theta = state[2][0]
	thetaArg = theta + (delTheta/2)
	state = state + [[delS*math.cos(thetaArg)], [delS*math.sin(thetaArg)], [delTheta]]
	return state

def calculatePredictionStepJacobians(delS, thetaArg, wheelbase=WHEELBASE):
	"""
	Calculates the Jacobians for the Prediction Step: partial derivative of configuration w.r.t state (x,y,z) (fp) and odometry (delSl, delSr) (fs)
    :param delS: 		control function inputs: tiny (hence del) distance travelled by imaginary point along robot's centerline
	:param thetaArg: 	a value equal to theta + (delTheta/2), needed for other calculation
	:param wheelbase: 	distance between differential-drive robot's two wheels
    :returns: 			jacobians
    """
	fp = numpy.array([[1, 0, -1*delS*math.sin(thetaArg)], [0, 1, delS*math.cos(thetaArg)], [0, 0, 1]]).reshape((3,3))
	fpT = fp.transpose().reshape((3,3))
	fs = numpy.array([[(0.5*math.cos(thetaArg))-(delS/(2*wheelbase))*math.sin(thetaArg), (0.5*math.cos(thetaArg))+((delS/(2*wheelbase))*math.sin(thetaArg))], [(0.5*math.sin(thetaArg))+(delS/(2*wheelbase))*math.cos(thetaArg), (0.5*math.sin(thetaArg))-((delS/(2*wheelbase))*math.cos(thetaArg))], [(1/wheelbase), (-1/wheelbase)]]).reshape((3,2))
	fsT = fs.transpose().reshape((2,3))	
	return fp, fpT, fs, fsT

def pose2DUncertainty(covarianceMatrix3x3):
	"""
	Calculates the overall uncertainty in the state estimate
	:param covarianceMatrix3x3: 3x3 the predicted covariance
	:returns:     				the overall uncertainty in the robot pose estimate
	"""
	if numpy.shape(covarianceMatrix3x3) != (3,3):
		return None
	determinant = ( covarianceMatrix3x3[0][0] * ( (covarianceMatrix3x3[1][1]*covarianceMatrix3x3[2][2]) - (covarianceMatrix3x3[2][1]*covarianceMatrix3x3[1][2]) )) - ( covarianceMatrix3x3[0][1] * ( (covarianceMatrix3x3[1][0]*covarianceMatrix3x3[2][2]) - (covarianceMatrix3x3[2][0]*covarianceMatrix3x3[1][2]) )) + ( covarianceMatrix3x3[0][2] * ( (covarianceMatrix3x3[1][0]*covarianceMatrix3x3[2][1]) - (covarianceMatrix3x3[1][1]*covarianceMatrix3x3[2][0]) ))
	return math.sqrt(abs(determinant))

# NOTE: tightly-coupled function: modifies 5, uses several other global variables
def LKMPredictionStep(odometry):
	# currentLeftWheelTicks = odometry['x']
	currentLeftWheelTicks = odometry.x
	# currentRightWheelTicks = odometry['y']
	currentRightWheelTicks = odometry.y
	global gPrevTotalLeftTicks, gPrevTotalRightTicks, gState, gP, gUncertainty

	# 1. estimate state

	leftTicksDifference = currentLeftWheelTicks - gPrevTotalLeftTicks
	rightTicksDifference = currentRightWheelTicks - gPrevTotalRightTicks
	
	# TODO: how to sample random (Gaussian) noise for encoder: del_sl += error_sl ?
	delSl = WHEEL_RADIUS * (leftTicksDifference * ANGLE_PER_TICK)
	errorSl = KL*abs(delSl)
	delSr = WHEEL_RADIUS * (rightTicksDifference * ANGLE_PER_TICK)
	errorSr = KR*abs(delSr)
	
	delS = (delSr + delSl) / 2
	delTheta = (delSr - delSl) / WHEELBASE
	theta = gState[2][0]
	thetaArg = theta + (delTheta/2)	

	gState = gState + [[delS*math.cos(thetaArg)], [delS*math.sin(thetaArg)], [delTheta]]
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
	
	gP = (fp @ gP @ fpT) + (fs @ odometryCovariance @ fsT)

	# overall uncertainty for pose (from Corke, pg. 160)
	robotPoseDeterminant = ( gP[0][0] * ( (gP[1][1]*gP[2][2]) - (gP[2][1]*gP[1][2]) )) - ( gP[0][1] * ( (gP[1][0]*gP[2][2]) - (gP[2][0]*gP[1][2]) )) + ( gP[0][2] * ( (gP[1][0]*gP[2][1]) - (gP[1][1]*gP[2][0]) ))
	gUncertainty = math.sqrt(abs(robotPoseDeterminant))

	gPrevTotalLeftTicks = currentLeftWheelTicks
	gPrevTotalRightTicks = currentRightWheelTicks

	# print("state: \n", state)
	# print("Covariance matrix: \n", p)
	# print("uncertainty: \n", uncertainty)

	print("From Prediction Step")
	pose2D = [gState[0][0], gState[1][0], gState[2][0]]
	covariance1x36 = numpy.append(numpy.reshape(gP, (1,9)), numpy.zeros(27))
	publishPoseWithCovariance(gPosePublisher, pose2D, covariance1x36)
	publishPoseWithCovariance(gEKFPredictionStepPosePublisher, pose2D, covariance1x36)




def updateStepCallback(beaconData):
	"""
    Callback function for messages received on /EncodersCubeBot/beacon_data topic.
	Performs the update step of EKF Localization (see updateStep() function); publishes the updated state estimate and covariance matrix; and performs several side effects (see comment-wrapped block below)
    """
	state, p = updateStep(gState, gP, beaconData)
	## [SIDE EFFECTS]
	gSetState(state)
	gSetStateCovariance(p)
	gSetUncertainty(p)
	pose2D = [state[0][0], state[1][0], state[2][0]]
	covariance1x36 = numpy.append(numpy.reshape(p, (1,9)), numpy.zeros(27))
	publishPoseWithCovariance(gPosePublisher, pose2D, covariance1x36)
	publishPoseWithCovariance(gEKFUpdateStepPosePublisher, pose2D, covariance1x36)
	## [SIDE EFFECTS]
	print("[update] state: \n", state)
	print("[update] p: \n", p)
	print("[update] uncertainty: ", gUncertainty)

def updateStep(state, p, beaconData):
	"""
    Performs the update step of EKF Localization: updates the estimate of the state, covariance, and the overall uncertainty (global variables: state, p, uncertainty) using new data (innovation)
    :param state: 		current state estimate
    :param p: 			covariance matrix for current state estimate
    :param beaconData:  { header={ stamp={ secs, nsecs }, frame_id }, point={ x (bearing [degrees]), y, z } })
    :returns:     		updates state estimate and covariance
    """
	robotX, robotY, robotTheta = ungroupStateNumpyArray(state)
	measuredBearing = beaconData.point.x	
	beaconX, beaconY = lookupBeaconPosition(gBeaconPositions, beaconData)
	innovation = calculateBeaconDataInnovation(state, measuredBearing, beaconX, beaconY)
	hw, hwT, hx, hxT = calculateUpdateStepJacobians(beaconX, beaconY, robotX, robotY)
	k = calculateKalmanGain(p, gBeaconDataCovariance, hw, hwT, hx, hxT)
	state = state + (k * innovation)  # 3x1 + (3x1 1x1) => 3x1
	p = p - (k @ hx @ p)
	return state, p
	
def lookupBeaconPosition(beaconPositionLookupTable, beaconData):
	"""
	Looks up the position of a beacon based on the beacon's (frame_)id
	:param beaconPositionLookupTable:	an object that contains the (frame_)id and positions for every beacon in the environment
	:param beaconData: 					{ header={ stamp={ secs, nsecs }, frame_id }, point={ x (bearing [degrees]), y, z } })
	:returns:							beacon position (x,y)
	"""
	# beaconX = beaconPositions[beaconData['header']['frame_id']]['x']
	beaconX = beaconPositionLookupTable[beaconData.header.frame_id]['x']
	beaconY = beaconPositionLookupTable[beaconData.header.frame_id]['y']
	return beaconX, beaconY

def calculateBeaconDataInnovation(state, measuredBearing, beaconX, beaconY):
	"""
    Calculates the innovation based on expected robot position and landmark (beacon) position
    :param state:   			robot state
    :param measuredBearing: 	bearing from beacon scanner reference frame to beacon
    :param beaconX, beaconY: 	known (x,y) position of beacon
    :returns:    				returns innovation (new information) from beacon data
    """
	robotX, robotY, robotTheta = ungroupStateNumpyArray(state)
	measuredBeaconData = numpy.array([measuredBearing]).reshape((1,1))	# radians
	expectedBeaconData = numpy.array([math.atan2((beaconY - robotY), (beaconX - robotX)) - robotTheta]).reshape((1,1))
	beaconDataNoise = numpy.array([gGetBearingNoiseSample()]).reshape((1,1))
	gIncrementBearingNoiseSampleIndex()
	return calculateInnovation(expectedBeaconData, measuredBeaconData, beaconDataNoise)

def calculateInnovation(expectedData, measuredData, noise):
	"""
	Calculates the innovation from expected and measured data
	"""
	innovation = measuredData - expectedData + noise
	return innovation

def calculateUpdateStepJacobians(landmarkX, landmarkY, robotX, robotY):
	"""
    Calculates the Jacobians for the update step: partial derivative of expected observation equation w.r.t state (x,y,z) (hx) and measurement (hw)
    :param landmarkX, landmarkY: 	x- and y-coordinates of landmark being used for the calculation
	:param robotX, robotY: 			x- and y-coordinates of robot
    :returns: 						jacobians hw, hwT, hx, hxT
    """
	distance = math.sqrt((landmarkX - robotX)**2 + (landmarkY - robotY)**2)
	hw = numpy.array([1]).reshape((1,1))  # hw = dh_bearing/dw_bearing => 1x1 scalar
	hwT = numpy.transpose(hw).reshape((1,1))
	hx = numpy.array([(landmarkY - robotY)/distance**2, -(landmarkX - robotX)/distance**2, -1]).reshape((1,3))  # 1x3 vector
	hxT = numpy.transpose(hx).reshape((3,1))
	return hw, hwT, hx, hxT

def calculateKalmanGain(stateCovariance, beaconDataCovariance, hw, hwT, hx, hxT):
	"""
	Calculates the Kalman gain for the update step of EKF localization
	:param stateCovariance:		state ovariance matrix
	:param hw, hwT, hx, hxT:	jacobians needed for calculation
	:return:					Kalman gain k
	"""
	s = (hx @ stateCovariance @ hxT) + (hw * beaconDataCovariance * hwT)  # 1x1 scalar
	sInverse = 1/s  # 1x1 scalar
	k = stateCovariance @ (hxT * sInverse)  # 3x3 3x1 1x1 => 3x3 3x1 => 3x1
	k = numpy.reshape(k, (3,1))
	return k

# beaconData: { header={ stamp={ secs, nsecs }, frame_id }, point={ x (bearing [degrees]), y, z } })
def LKMUpdateStepWithBeaconData(beaconData):
	global gState, gP, gUncertainty, gBearingNoiseSamplesIndex

	# 3. calculate innovation: measured angle - expected angle + sensor noise

	# measuredBearing = beaconData['point']['x']
	measuredBearing = beaconData.point.x	
	# beaconX = beaconPositions[beaconData['header']['frame_id']]['x']
	beaconX = gBeaconPositions[beaconData.header.frame_id]['x']
	# beaconY = beaconPositions[beaconData['header']['frame_id']]['y']
	beaconY = gBeaconPositions[beaconData.header.frame_id]['y']
	robotX = gState[0][0]
	robotY = gState[1][0]
	robotTheta = gState[2][0]
	expectedBeaconRange = math.sqrt((beaconX - robotX)**2 + (beaconY - robotY)**2)

	measuredBeaconData = numpy.array([measuredBearing]).reshape((1,1))	# radians
	expectedBeaconData = numpy.array([math.atan2((beaconY - robotY), (beaconX - robotX)) - robotTheta]).reshape((1,1))
	
	beaconDataNoise = numpy.array([gBearingNoiseSamples[gBearingNoiseSamplesIndex]]).reshape((1,1))
	gBearingNoiseSamplesIndex = 0 if gBearingNoiseSamplesIndex == len(gBearingNoiseSamples)-1 else gBearingNoiseSamplesIndex + 1
	innovation = measuredBeaconData - expectedBeaconData + beaconDataNoise


	# 4. update state estimate

	# hw = numpy.array([[1, 0], [0, 1]])
	print("p: \n", gP)
	print("beaconDataCovariance: \n", gBeaconDataCovariance)

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
	s = (hx @ gP @ hxT) + (hw * gBeaconDataCovariance * hwT)  # 1x1 scalar
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

	k = gP @ (hxT * sInverse)  # 3x3 3x1 1x1 => 3x3 3x1 => 3x1
	k = numpy.reshape(k, (3,1))
	print("k: ", numpy.shape(k))
	print(k)
	
	gState = gState + (k * innovation)  # 3x1 + (3x1 1x1) => 3x1
	print("state: \n", gState)

	# 5. update covariance
	# psub = k @ (hx @ p)
	# psub = numpy.reshape(psub, (3,3))
	# print("psub: \n", psub)

	gP = gP - (k @ hx @ gP)
	print("p: \n", gP)

	robotPoseDeterminant = ( gP[0][0] * ( (gP[1][1]*gP[2][2]) - (gP[2][1]*gP[1][2]) )) - ( gP[0][1] * ( (gP[1][0]*gP[2][2]) - (gP[2][0]*gP[1][2]) )) + ( gP[0][2] * ( (gP[1][0]*gP[2][1]) - (gP[1][1]*gP[2][0]) ))
	gUncertainty = math.sqrt(abs(robotPoseDeterminant))

	print("From Update Step")
	pose2D = [gState[0][0], gState[1][0], gState[2][0]]
	covariance1x36 = numpy.append(numpy.reshape(gP, (1,9)), numpy.zeros(27))
	print("pose2D: \n", pose2D)
	print("covariance1x36: \n", covariance1x36)
	publishPoseWithCovariance(gPosePublisher, pose2D, covariance1x36)
	publishPoseWithCovariance(gEKFUpdateStepPosePublisher, pose2D, covariance1x36)




def publishPoseWithCovariance(publisher, pose2D, covariance1D):
	# LESSON: 
	# -> initialize a message type and then set each property using its full dot notation name
	#    doing this will prevent you from getting a bunch of "AttributeError: 'dict' object has no attribute 'attribute'" errors
	poseStamped = PoseWithCovarianceStamped()
	qx, qy, qz, qw = eulerToQuaternion(0, 0, round(pose2D[2]))	
	poseStamped.pose.pose.position.x = round(pose2D[0], 3)
	poseStamped.pose.pose.position.y = round(pose2D[1], 3)
	poseStamped.pose.pose.position.z = 0
	poseStamped.pose.pose.orientation.x = qx
	poseStamped.pose.pose.orientation.y = qy
	poseStamped.pose.pose.orientation.z = qz
	poseStamped.pose.pose.orientation.w = qw
	poseStamped.pose.covariance = covariance1D.tolist()
	publisher.publish(poseStamped)

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