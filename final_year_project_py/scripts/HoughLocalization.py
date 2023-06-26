"""
@file HoughLocalization.py
@brief This program finds the (real and virtual) intersections between lines 
derived from the walls of an environment (detected via Hough Transform) 
for use as landmarks in robot localization.

Relevant links:
    https://docs.opencv.org/3.4/d3/de6/tutorial_js_houghlines.html
    https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
"""


from __future__ import print_function
import cv2 as cv
from PIL import Image as im
import sys, rospy, math, numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Point


IMAGE_WIDTH = 150*2
IMAGE_HEIGHT = 150*2

landmarkPositionPublisher = rospy.Publisher('/HoughLocalization/landmark_positions', Point, queue_size=10)  


def cartXY2imageXY(cartX, cartY, image_width, image_height):
    imageX = int(cartX + image_width/2)
    imageY = int(image_height/2 - cartY)
    return imageX, imageY

def plotRangeAndBearingDataOnImage(rangeDataInM, totalSweepAngleInRad, imageWidth, imageHeight, showImage=True):
    MONO_BLACK = 0
    MONO_WHITE = 255
    imageArray = [[MONO_WHITE for i in range(imageWidth)] for j in range(imageHeight)]  # order: [row][col] = [y][x]
    
    # convert range and bearing data from polar to Cartesian coords.
    sweepAngleInRad = totalSweepAngleInRad/len(rangeDataInM)
    bearingDataInRad = [sweepAngleInRad*i for i in range(len(rangeDataInM))]
    x = [abs(rangeDataInM[i])*math.cos(bearingDataInRad[i]) for i in range(len(rangeDataInM))]
    y = [abs(rangeDataInM[i])*math.sin(bearingDataInRad[i]) for i in range(len(rangeDataInM))]

    # plot Cartesian coords. on image
	# convert scale to cm; easier to work with
    meterToCMMultiplier = 100
    for i in range(len(rangeDataInM)):
        imageX, imageY = cartXY2imageXY(x[i]*meterToCMMultiplier, y[i]*meterToCMMultiplier, imageWidth, imageHeight)
        imageArray[imageY][imageX] = MONO_BLACK
        
    numPyImageArray = np.array(imageArray, np.uint8)
    image = im.fromarray(numPyImageArray)
    if showImage == True:
        cv.imshow("Range and Bearing Data", numPyImageArray)
        cv.waitKey()
    return image

def HoughLocalization():
    # ## [rangefinder_readings_to_image]
    # noObstaclesRawData = [ 0.864, 0.864, 0.865, 0.865, 0.866, 0.867, 0.868, 0.869, 0.871, 0.872, 0.874, 0.876, 0.878, 0.881, 0.883, 0.886, 0.889, 0.892, 0.895, 0.899, 0.902, 0.906, 0.883, 0.848, 0.815, 0.785, 0.757, 0.731, 0.707, 0.685, 0.664, 0.645, 0.627, 0.61, 0.594, 0.579, 0.564, 0.551, 0.538, 0.526, 0.514, 0.504, 0.493, 0.483, 0.474, 0.465, 0.457, 0.449, 0.441, 0.438, 0.443, 0.448, 0.454, 0.46, 0.466, 0.472, 0.479, 0.486, 0.493, 0.501, 0.509, 0.517, 0.526, 0.535, 0.545, 0.554, 0.565, 0.576, 0.587, 0.599, 0.612, 0.626, 0.64, 0.655, 1.045, 1.037, 1.029, 1.021, 1.014, 1.007, 1, 0.994, 0.987, 0.982, 0.976, 0.971, 0.965, 0.96, 0.956, 0.951, 0.947, 0.943, 0.939, 0.936, 0.933, 0.929, 0.926, 0.924, 0.921, 0.919, 0.917, 0.915, 0.913, 0.911, 0.91, 0.909, 0.907, 0.907, 0.906, 0.905, 0.905, 0.905, 0.905, 0.905, 0.905, 0.906, 0.907, 0.907, 0.909, 0.91, 0.911, 0.913, 0.915, 0.917, 0.919, 0.921, 0.924, 0.926, 0.929, 0.933, 0.936, 0.939, 0.943, 0.947, 0.951, 0.956, 0.96, 0.965, 0.971, 0.943, 0.911, 0.882, 0.855, 0.83, 0.806, 0.783, 0.762, 0.743, 0.724, 0.706, 0.69, 0.674, 0.659, 0.645, 0.632, 0.619, 0.607, 0.595, 0.584, 0.574, 0.564, 0.554, 0.545, 0.536, 0.528, 0.52, 0.512, 0.505, 0.498, 0.491, 0.485, 0.478, 0.472, 0.467, 0.461, 0.456, 0.451, 0.446, 0.441, 0.437, 0.432, 0.428, 0.424, 0.42, 0.417, 0.413, 0.41, 0.406, 0.403, 0.4, 0.397, 0.394, 0.392, 0.389, 0.387, 0.384, 0.382, 0.38, 0.378, 0.376, 0.374, 0.372, 0.371, 0.369, 0.367, 0.366, 0.365, 0.363, 0.362, 0.361, 0.36, 0.359, 0.358, 0.357, 0.357, 0.356, 0.355, 0.355, 0.354, 0.354, 0.354, 0.354, 0.353, 0.353, 0.353, -0.355, -0.355, -0.355, -0.355, -0.356, -0.356, -0.357, -0.357, -0.358, -0.358, -0.359, -0.36, -0.361, -0.362, -0.363, -0.364, -0.365, -0.366, -0.368, -0.369, -0.371, -0.372, -0.374, -0.376, -0.378, -0.38, -0.382, -0.384, -0.386, -0.389, -0.391, -0.394, -0.396, -0.399, -0.402, -0.405, -0.408, -0.412, -0.415, -0.419, -0.422, -0.426, -0.43, -0.435, -0.439, -0.443, -0.448, -0.453, -0.458, -0.464, -0.469, -0.471, -0.464, -0.457, -0.451, -0.444, -0.438, -0.432, -0.427, -0.421, -0.416, -0.411, -0.406, -0.402, -0.397, -0.393, -0.389, -0.385, -0.381, -0.378, -0.374, -0.371, -0.368, -0.365, -0.362, -0.359, -0.356, -0.353, -0.351, -0.348, -0.346, -0.344, -0.342, -0.34, -0.338, -0.336, -0.334, -0.332, -0.331, -0.329, -0.328, -0.326, -0.325, -0.324, -0.323, -0.322, -0.321, -0.32, -0.319, -0.318, -0.317, -0.317, -0.316, -0.315, -0.315, -0.314, -0.314, -0.314, -0.314, -0.313, -0.313, -0.313, -0.313, -0.313, -0.313, -0.314, -0.314, -0.314, -0.315, -0.315, -0.315, -0.316, -0.317, -0.317, -0.318, -0.319, -0.32, -0.321, -0.322, -0.323, -0.324, -0.325, -0.327, -0.328, -0.329, -0.331, -0.333, -0.334, -0.336, -0.338, -0.34, -0.342, -0.344, -0.346, -0.349, -0.351, -0.354, -0.356, -0.359, -0.362, -0.365, -0.368, -0.371, -0.374, -0.378, -0.382, -0.385, -0.389, -0.393, -0.398, -0.402, -0.407, -0.41, -0.415, -0.42, -0.426, -0.433, -0.439, -0.445, -0.451, -0.458, -0.465, -0.472, -0.48, -0.488, -0.496, -0.505, -0.514, -0.523, -0.533, -0.544, -0.555, -0.566, -0.579, -0.592, -0.605, -0.62, -0.635, -0.651, -0.668, -0.686, -0.705, -0.726, -0.748, -0.771, -0.796, -0.823, -0.852, -0.883, -0.917, -0.913, -0.909, -0.904, -0.901, -0.897, -0.893, -0.89, -0.887, -0.884, -0.881, -0.879, -0.877, -0.874, -0.872, -0.871, -0.869, -0.868, -0.866, -0.865, -0.864, -0.864, -0.863, -0.863, -0.862, -0.862 ];
    # withObstaclesRawData = [0.864, 0.864, 0.865, 0.865, 0.866, 0.867, 0.868, 0.869, 0.871, 0.872, 0.874, 0.876, 0.878, 0.881, 0.883, 0.886, 0.889, 0.892, 0.895, 0.899, 0.902, 0.906, 0.883, 0.848, 0.815, 0.785, 0.476, 0.473, 0.452, 0.459, 0.456, 0.454, 0.443, 0.453, 0.455, 0.458, 0.453, 0.456, 0.477, 0.478, 0.514, 0.504, 0.493, 0.483, 0.474, 0.465, 0.457, 0.449, 0.441, 0.438, 0.443, 0.448, 0.454, 0.46, 0.466, 0.472, 0.479, 0.486, 0.493, 0.501, 0.509, 0.517, 0.526, 0.535, 0.545, 0.554, 0.565, 0.576, 0.587, 0.599, 0.612, 0.626, 0.64, 0.655, 1.045, 1.037, 1.029, 1.021, 1.014, 1.007, 1, 0.994, 0.987, 0.982, 0.976, 0.971, 0.965, 0.96, 0.956, 0.951, 0.947, 0.943, 0.939, 0.936, 0.933, 0.929, 0.926, 0.924, 0.921, 0.919, 0.917, 0.915, 0.913, 0.911, 0.91, 0.909, 0.907, 0.907, 0.906, 0.905, 0.905, 0.905, 0.905, 0.905, 0.905, 0.906, 0.907, 0.907, 0.909, 0.91, 0.911, 0.913, 0.915, 0.917, 0.919, 0.921, 0.924, 0.798, 0.784, 0.759, 0.765, 0.762, 0.763, 0.76, 0.777, 0.799, 0.96, 0.965, 0.971, 0.943, 0.911, 0.882, 0.855, 0.83, 0.806, 0.783, 0.762, 0.743, 0.724, 0.706, 0.69, 0.674, 0.659, 0.312, 0.305, 0.299, 0.293, 0.288, 0.283, 0.277, 0.273, 0.268, 0.264, 0.259, 0.255, 0.251, 0.248, 0.244, 0.246, 0.249, 0.253, 0.257, 0.261, 0.265, 0.269, 0.274, 0.278, 0.284, 0.289, 0.294, 0.3, 0.306, 0.313, 0.42, 0.417, 0.413, 0.41, 0.406, 0.403, 0.4, 0.397, 0.394, 0.392, 0.389, 0.387, 0.384, 0.382, 0.38, 0.378, 0.376, 0.374, 0.372, 0.371, 0.369, 0.367, 0.366, 0.365, 0.363, 0.362, 0.361, 0.36, 0.359, 0.358, 0.357, 0.357, 0.356, 0.355, 0.355, 0.354, 0.354, 0.354, 0.354, 0.353, 0.353, 0.353, -0.355, -0.355, -0.355, -0.355, -0.356, -0.356, -0.357, -0.357, -0.358, -0.358, -0.359, -0.36, -0.361, -0.362, -0.363, -0.364, -0.365, -0.366, -0.368, -0.369, -0.371, -0.372, -0.374, -0.376, -0.378, -0.38, -0.382, -0.384, -0.386, -0.389, -0.391, -0.394, -0.396, -0.399, -0.402, -0.405, -0.408, -0.412, -0.415, -0.419, -0.422, -0.426, -0.43, -0.435, -0.439, -0.443, -0.448, -0.453, -0.458, -0.464, -0.469, -0.471, -0.464, -0.457, -0.451, -0.444, -0.438, -0.432, -0.427, -0.421, -0.416, -0.411, -0.406, -0.402, -0.397, -0.393, -0.389, -0.385, -0.381, -0.378, -0.374, -0.371, -0.368, -0.365, -0.362, -0.359, -0.356, -0.353, -0.351, -0.348, -0.346, -0.344, -0.342, -0.34, -0.338, -0.336, -0.334, -0.332, -0.331, -0.329, -0.328, -0.326, -0.325, -0.324, -0.323, -0.322, -0.321, -0.32, -0.319, -0.318, -0.317, -0.317, -0.316, -0.315, -0.315, -0.314, -0.314, -0.314, -0.314, -0.313, -0.313, -0.313, -0.313, -0.313, -0.313, -0.314, -0.314, -0.314, -0.315, -0.315, -0.315, -0.316, -0.317, -0.317, -0.318, -0.319, -0.32, -0.321, -0.322, -0.323, -0.26, -0.247, -0.235, -0.225, -0.215, -0.206, -0.198, -0.191, -0.184, -0.184, -0.185, -0.186, -0.187, -0.188, -0.189, -0.191, -0.192, -0.193, -0.195, -0.197, -0.198, -0.2, -0.202, -0.203, -0.205, -0.207, -0.209, -0.211, -0.214, -0.216, -0.218, -0.221, -0.223, -0.226, -0.229, -0.426, -0.433, -0.439, -0.445, -0.451, -0.458, -0.465, -0.472, -0.48, -0.488, -0.496, -0.505, -0.514, -0.523, -0.533, -0.544, -0.555, -0.566, -0.579, -0.592, -0.605, -0.62, -0.635, -0.651, -0.668, -0.686, -0.705, -0.726, -0.748, -0.771, -0.796, -0.823, -0.852, -0.883, -0.917, -0.787, -0.779, -0.758, -0.762, -0.76, -0.752, -0.766, -0.776, -0.775, -0.881, -0.879, -0.877, -0.874, -0.872, -0.871, -0.869, -0.868, -0.866, -0.865, -0.864, -0.864, -0.863, -0.863, -0.862, -0.862];
    # rangeDataRawTest1 = [ 0.864, 0.864, 0.865, 0.865, 0.866, 0.867, 0.868, 0.869, 0.871, 0.472, 0.473, 0.474, 0.476, 0.455, 0.455, 0.456, 0.458, 0.448, 0.437, 0.439, 0.441, 0.442, 0.426, 0.422, 0.424, 0.427, 0.757, 0.731, 0.707, 0.685, 0.664, 0.645, 0.627, 0.61, 0.594, 0.579, 0.564, 0.551, 0.538, 0.526, 0.514, 0.504, 0.493, 0.483, 0.474, 0.465, 0.457, 0.449, 0.441, 0.438, 0.443, 0.448, 0.454, 0.46, 0.466, 0.472, 0.479, 0.486, 0.493, 0.501, 0.509, 0.517, 0.526, 0.535, 0.545, 0.554, 0.565, 0.576, 0.587, 0.599, 1.081, 1.072, 1.062, 1.053, 1.045, 1.037, 0.326, 0.323, 0.321, 0.319, 0.317, 0.314, 0.313, 0.311, 0.309, 0.312, 0.323, 0.336, 0.35, 0.348, 0.347, 0.345, 0.344, 0.343, 0.342, 0.34, 0.339, 0.338, 0.361, 0.919, 0.917, 0.915, 0.913, 0.911, 0.91, 0.909, 0.907, 0.907, 0.906, 0.905, 0.905, 0.905, 0.905, 0.905, 0.905, 0.906, 0.907, 0.907, 0.909, 0.91, 0.911, 0.913, 0.915, 0.917, 0.919, 0.921, 0.924, 0.926, 0.929, 0.268, 0.254, 0.247, 0.248, 0.249, 0.25, 0.251, 0.253, 0.236, 0.235, 0.233, 0.232, 0.231, 0.229, 0.228, 0.227, 0.226, 0.225, 0.224, 0.223, 0.23, 0.242, 0.238, 0.233, 0.228, 0.223, 0.219, 0.219, 0.221, 0.223, 0.226, 0.228, 0.231, 0.234, 0.536, 0.528, 0.52, 0.512, 0.505, 0.498, 0.491, 0.485, 0.478, 0.472, 0.467, 0.461, 0.456, 0.451, 0.446, 0.441, 0.437, 0.432, 0.428, 0.424, 0.42, 0.417, 0.413, 0.41, 0.406, 0.403, 0.4, 0.397, 0.394, 0.392, 0.389, 0.387, 0.384, 0.382, 0.38, 0.378, 0.376, 0.374, 0.372, 0.371, 0.369, 0.367, 0.366, 0.365, 0.363, 0.362, 0.361, 0.36, 0.359, 0.358, 0.357, 0.357, 0.356, 0.355, 0.355, 0.354, 0.354, 0.354, 0.354, 0.353, 0.353, 0.353, -0.355, -0.355, -0.355, -0.355, -0.356, -0.356, -0.357, -0.357, -0.358, -0.358, -0.359, -0.36, -0.361, -0.362, -0.363, -0.364, -0.365, -0.366, -0.368, -0.369, -0.371, -0.372, -0.374, -0.376, -0.378, -0.38, -0.17, -0.164, -0.159, -0.154, -0.149, -0.15, -0.151, -0.152, -0.153, -0.154, -0.155, -0.157, -0.158, -0.145, -0.144, -0.144, -0.143, -0.143, -0.143, -0.142, -0.142, -0.142, -0.142, -0.142, -0.141, -0.141, -0.141, -0.141, -0.141, -0.154, -0.152, -0.15, -0.148, -0.146, -0.144, -0.143, -0.141, -0.142, -0.145, -0.147, -0.15, -0.153, -0.156, -0.159, -0.157, -0.156, -0.154, -0.153, -0.152, -0.151, -0.15, -0.148, -0.147, -0.151, -0.155, -0.16, -0.165, -0.34, -0.338, -0.336, -0.334, -0.332, -0.331, -0.329, -0.328, -0.326, -0.325, -0.324, -0.323, -0.322, -0.321, -0.32, -0.319, -0.318, -0.317, -0.317, -0.316, -0.315, -0.315, -0.314, -0.314, -0.314, -0.314, -0.313, -0.313, -0.313, -0.313, -0.313, -0.313, -0.314, -0.314, -0.314, -0.315, -0.315, -0.315, -0.316, -0.317, -0.317, -0.318, -0.319, -0.32, -0.321, -0.322, -0.323, -0.324, -0.325, -0.327, -0.328, -0.329, -0.331, -0.333, -0.334, -0.336, -0.338, -0.34, -0.342, -0.344, -0.346, -0.349, -0.351, -0.354, -0.356, -0.359, -0.362, -0.365, -0.368, -0.371, -0.374, -0.378, -0.382, -0.385, -0.389, -0.393, -0.398, -0.402, -0.407, -0.41, -0.415, -0.42, -0.426, -0.433, -0.439, -0.445, -0.451, -0.458, -0.465, -0.472, -0.48, -0.488, -0.496, -0.505, -0.514, -0.523, -0.533, -0.544, -0.555, -0.566, -0.579, -0.592, -0.605, -0.62, -0.635, -0.651, -0.327, -0.324, -0.322, -0.32, -0.318, -0.316, -0.323, -0.334, -0.343, -0.341, -0.339, -0.337, -0.336, -0.334, -0.337, -0.353, -0.361, -0.36, -0.359, -0.357, -0.356, -0.355, -0.373, -0.874, -0.872, -0.871, -0.869, -0.868, -0.866, -0.865, -0.864, -0.864, -0.863, -0.863, -0.862, -0.862 ];
    # rangeDataRawTest2 = [ 0.864, 0.864, 0.865, 0.865, 0.866, 0.7, 0.701, 0.687, 0.688, 0.683, 0.874, 0.669, 0.662, 0.881, 0.639, 0.641, 0.627, 0.63, 0.895, 0.899, 0.902, 0.906, 0.883, 0.848, 0.815, 0.785, 0.757, 0.731, 0.707, 0.685, 0.664, 0.645, 0.627, 0.61, 0.594, 0.579, 0.564, 0.551, 0.538, 0.526, 0.514, 0.504, 0.493, 0.483, 0.474, 0.465, 0.457, 0.449, 0.441, 0.438, 0.443, 0.448, 0.454, 0.46, 0.466, 0.472, 0.479, 0.486, 0.493, 0.501, 0.509, 0.517, 0.526, 0.535, 0.545, 0.554, 0.565, 0.576, 0.587, 0.599, 1.081, 1.072, 1.062, 1.053, 1.045, 1.037, 0.326, 0.323, 0.321, 0.319, 0.317, 0.314, 0.313, 0.311, 0.309, 0.312, 0.323, 0.336, 0.35, 0.348, 0.347, 0.345, 0.344, 0.343, 0.342, 0.34, 0.339, 0.338, 0.361, 0.919, 0.917, 0.915, 0.913, 0.911, 0.91, 0.909, 0.907, 0.907, 0.906, 0.905, 0.905, 0.905, 0.905, 0.905, 0.905, 0.906, 0.907, 0.907, 0.909, 0.91, 0.911, 0.913, 0.915, 0.917, 0.919, 0.921, 0.924, 0.926, 0.929, 0.268, 0.254, 0.247, 0.248, 0.249, 0.25, 0.251, 0.253, 0.236, 0.235, 0.233, 0.232, 0.231, 0.229, 0.228, 0.227, 0.226, 0.225, 0.224, 0.223, 0.23, 0.242, 0.238, 0.233, 0.228, 0.223, 0.219, 0.219, 0.221, 0.223, 0.226, 0.228, 0.231, 0.234, 0.536, 0.528, 0.52, 0.512, 0.505, 0.498, 0.491, 0.485, 0.478, 0.472, 0.467, 0.461, 0.456, 0.451, 0.446, 0.441, 0.437, 0.432, 0.428, 0.424, 0.42, 0.417, 0.413, 0.41, 0.406, 0.403, 0.4, 0.397, 0.394, 0.392, 0.389, 0.387, 0.384, 0.382, 0.38, 0.378, 0.376, 0.374, 0.372, 0.371, 0.369, 0.367, 0.366, 0.365, 0.363, 0.362, 0.361, 0.36, 0.359, 0.358, 0.357, 0.357, 0.356, 0.355, 0.355, 0.354, 0.354, 0.354, 0.354, 0.353, 0.353, 0.353, -0.355, -0.355, -0.355, -0.355, -0.356, -0.356, -0.357, -0.357, -0.358, -0.358, -0.359, -0.36, -0.361, -0.362, -0.363, -0.364, -0.365, -0.366, -0.368, -0.369, -0.371, -0.372, -0.374, -0.172, -0.165, -0.159, -0.16, -0.161, -0.162, -0.163, -0.164, -0.165, -0.161, -0.157, -0.152, -0.153, -0.154, -0.155, -0.157, -0.158, -0.159, -0.426, -0.43, -0.435, -0.439, -0.443, -0.448, -0.157, -0.157, -0.157, -0.156, -0.156, -0.156, -0.156, -0.156, -0.156, -0.438, -0.432, -0.427, -0.421, -0.416, -0.411, -0.157, -0.157, -0.157, -0.157, -0.158, -0.385, -0.381, -0.159, -0.157, -0.156, -0.154, -0.153, -0.152, -0.152, -0.156, -0.16, -0.164, -0.163, -0.162, -0.161, -0.16, -0.159, -0.158, -0.163, -0.169, -0.332, -0.331, -0.329, -0.328, -0.326, -0.325, -0.324, -0.323, -0.322, -0.321, -0.32, -0.319, -0.318, -0.317, -0.317, -0.316, -0.315, -0.315, -0.314, -0.314, -0.314, -0.314, -0.313, -0.313, -0.313, -0.313, -0.313, -0.313, -0.314, -0.314, -0.314, -0.315, -0.315, -0.315, -0.316, -0.317, -0.317, -0.318, -0.319, -0.32, -0.321, -0.322, -0.323, -0.324, -0.325, -0.327, -0.328, -0.329, -0.331, -0.333, -0.334, -0.336, -0.338, -0.34, -0.342, -0.344, -0.346, -0.349, -0.351, -0.354, -0.356, -0.359, -0.362, -0.365, -0.368, -0.371, -0.374, -0.378, -0.382, -0.385, -0.389, -0.393, -0.398, -0.402, -0.407, -0.41, -0.415, -0.42, -0.426, -0.433, -0.439, -0.445, -0.451, -0.458, -0.465, -0.472, -0.48, -0.488, -0.496, -0.505, -0.514, -0.523, -0.533, -0.544, -0.555, -0.566, -0.579, -0.592, -0.605, -0.62, -0.635, -0.651, -0.327, -0.324, -0.322, -0.32, -0.318, -0.316, -0.323, -0.334, -0.343, -0.341, -0.339, -0.337, -0.336, -0.334, -0.337, -0.353, -0.361, -0.36, -0.359, -0.357, -0.356, -0.355, -0.373, -0.874, -0.872, -0.871, -0.869, -0.868, -0.866, -0.865, -0.864, -0.864, -0.863, -0.863, -0.862, -0.862 ];
    # morseLandmarkTestNoNoise = [0.644000, 0.644000, 0.645000, 0.645000, 0.646000, 0.646000, 0.647000, 0.648000, 0.649000, 0.650000, 0.652000, 0.653000, 0.655000, 0.657000, 0.658000, 0.660000, 0.663000, 0.665000, 0.667000, 0.670000, 0.673000, 0.676000, 0.679000, 0.682000, 0.685000, 0.689000, 0.693000, 0.697000, 0.701000, 0.705000, 0.709000, 0.714000, 0.719000, 0.724000, 0.729000, 0.735000, 0.741000, 0.747000, 0.753000, 0.320000, 0.313000, 0.310000, 0.313000, 0.316000, 0.314000, 0.308000, 0.307000, 0.310000, 0.313000, 0.841000, 0.851000, 0.861000, 0.853000, 0.313000, 0.313000, 0.313000, 0.313000, 0.313000, 0.784000, 0.774000, 0.764000, 0.755000, 0.746000, 0.311000, 0.308000, 0.306000, 0.312000, 0.317000, 0.314000, 0.311000, 0.310000, 0.317000, 0.675000, 0.669000, 0.664000, 0.659000, 0.654000, 0.649000, 0.644000, 0.640000, 0.635000, 0.631000, 0.627000, 0.624000, 0.620000, 0.617000, 0.613000, 0.610000, 0.607000, 0.604000, 0.602000, 0.599000, 0.597000, 0.595000, 0.592000, 0.590000, 0.589000, 0.587000, 0.585000, 0.584000, 0.582000, 0.581000, 0.580000, 0.579000, 0.578000, 0.577000, 0.577000, 0.576000, 0.576000, 0.575000, 0.575000, 0.575000, 0.575000, 0.575000, 0.575000, 0.576000, 0.576000, 0.577000, 0.577000, 0.578000, 0.579000, 0.580000, 0.581000, 0.582000, 0.584000, 0.585000, 0.587000, 0.589000, 0.590000, 0.592000, 0.595000, 0.597000, 0.599000, 0.602000, 0.604000, 0.607000, 0.610000, 0.613000, 0.616000, 0.620000, 0.623000, 0.627000, 0.631000, 0.635000, 0.639000, 0.644000, 0.649000, 0.653000, 0.658000, 0.664000, 0.669000, 0.675000, 0.681000, 0.687000, 0.693000, 0.700000, 0.707000, 0.465000, 0.460000, 0.465000, 0.465000, 0.457000, 0.461000, 0.466000, 0.466000, 0.465000, 0.794000, 0.804000, 0.808000, 0.797000, 0.465000, 0.465000, 0.466000, 0.461000, 0.456000, 0.464000, 0.465000, 0.460000, 0.465000, 0.708000, 0.701000, 0.695000, 0.688000, 0.682000, 0.676000, 0.670000, 0.664000, 0.659000, 0.654000, 0.649000, 0.644000, 0.640000, 0.635000, 0.631000, 0.627000, 0.623000, 0.620000, 0.616000, 0.613000, 0.610000, 0.607000, 0.604000, 0.601000, 0.599000, 0.596000, 0.594000, 0.592000, 0.590000, 0.588000, 0.586000, 0.584000, 0.583000, 0.581000, 0.580000, 0.579000, 0.578000, 0.577000, 0.576000, 0.575000, 0.574000, 0.574000, 0.574000, 0.573000, 0.573000, 0.573000, 0.574000, 0.574000, 0.574000, 0.574000, 0.575000, 0.576000, 0.576000, 0.577000, 0.578000, 0.579000, 0.580000, 0.582000, 0.583000, 0.585000, 0.586000, 0.588000, 0.590000, 0.592000, 0.594000, 0.597000, 0.599000, 0.602000, 0.604000, 0.607000, 0.610000, 0.613000, 0.617000, 0.620000, 0.624000, 0.628000, 0.632000, 0.636000, 0.640000, 0.645000, 0.649000, 0.654000, 0.660000, 0.665000, 0.670000, 0.676000, 0.682000, 0.688000, 0.695000, 0.702000, 0.709000, 0.716000, 0.724000, 0.624000, 0.614000, 0.621000, 0.616000, 0.616000, 0.622000, 0.621000, 0.621000, 0.621000, 0.621000, 0.621000, 0.621000, 0.620000, 0.612000, 0.621000, 0.617000, 0.618000, 0.816000, 0.808000, 0.799000, 0.791000, 0.783000, 0.776000, 0.769000, 0.762000, 0.755000, 0.749000, 0.743000, 0.737000, 0.731000, 0.726000, 0.721000, 0.716000, 0.711000, 0.706000, 0.702000, 0.698000, 0.694000, 0.690000, 0.686000, 0.683000, 0.680000, 0.676000, 0.673000, 0.671000, 0.668000, 0.665000, 0.663000, 0.661000, 0.659000, 0.657000, 0.655000, 0.653000, 0.652000, 0.650000, 0.649000, 0.648000, 0.647000, 0.646000, 0.645000, 0.644000, 0.644000, 0.644000, 0.643000, 0.643000, 0.643000, 0.643000, 0.644000, 0.644000, 0.644000, 0.645000, 0.645000, 0.647000, 0.648000, 0.649000, 0.650000, 0.652000, 0.653000, 0.655000, 0.657000, 0.659000, 0.661000, 0.663000, 0.665000, 0.668000, 0.670000, 0.673000, 0.676000, 0.679000, 0.683000, 0.686000, 0.167000, 0.161000, 0.159000, 0.160000, 0.161000, 0.162000, 0.163000, 0.164000, 0.163000, 0.159000, 0.155000, 0.152000, 0.153000, 0.155000, 0.156000, 0.157000, 0.159000, 0.783000, 0.791000, 0.799000, 0.807000, 0.816000, 0.825000, 0.835000, 0.844000, 0.855000, 0.865000, 0.876000, 0.888000, 0.900000, 0.907000, 0.895000, 0.883000, 0.872000, 0.861000, 0.850000, 0.840000, 0.831000, 0.821000, 0.812000, 0.804000, 0.796000, 0.788000, 0.780000, 0.773000, 0.159000, 0.157000, 0.156000, 0.155000, 0.154000, 0.153000, 0.157000, 0.161000, 0.166000, 0.165000, 0.164000, 0.163000, 0.162000, 0.161000, 0.160000, 0.165000, 0.171000, 0.682000, 0.678000, 0.675000, 0.672000, 0.670000, 0.667000, 0.665000, 0.662000, 0.660000, 0.658000, 0.656000, 0.654000, 0.653000, 0.651000, 0.650000, 0.649000, 0.648000, 0.647000, 0.646000, 0.645000, 0.645000, 0.644000, 0.644000, 0.644000, 0.644000]
    # morseLandmarkTestWithNoise = [0.645360, 0.646040, 0.646700, 0.663020, 0.635800, 0.629340, 0.643260, 0.644260, 0.659200, 0.655780, 0.648260, 0.657760, 0.660440, 0.652920, 0.647460, 0.665100, 0.657220, 0.676900, 0.690800, 0.678160, 0.680140, 0.675320, 0.685120, 0.671460, 0.677520, 0.683220, 0.700140, 0.685100, 0.688760, 0.691740, 0.717840, 0.710600, 0.729200, 0.713460, 0.728660, 0.744860, 0.738620, 0.771140, 0.768300, 0.314220, 0.314020, 0.288920, 0.327620, 0.313280, 0.343920, 0.302560, 0.304960, 0.306600, 0.316740, 0.839980, 0.846240, 0.871200, 0.866600, 0.335440, 0.304160, 0.308920, 0.311980, 0.321160, 0.800320, 0.771280, 0.761960, 0.750920, 0.743620, 0.320860, 0.298820, 0.295120, 0.317100, 0.291500, 0.298360, 0.309640, 0.315780, 0.304760, 0.688600, 0.680220, 0.650400, 0.666140, 0.658760, 0.648660, 0.633120, 0.626400, 0.650300, 0.628280, 0.630060, 0.622300, 0.631220, 0.609860, 0.603820, 0.618840, 0.613800, 0.598220, 0.616620, 0.582000, 0.587480, 0.608600, 0.624980, 0.593060, 0.596480, 0.602640, 0.595200, 0.602020, 0.594580, 0.577260, 0.578300, 0.572540, 0.576300, 0.574960, 0.566800, 0.586540, 0.582120, 0.578060, 0.598800, 0.561400, 0.601520, 0.583840, 0.583840, 0.571580, 0.593680, 0.593660, 0.577340, 0.585140, 0.577640, 0.577620, 0.585760, 0.608860, 0.595900, 0.595200, 0.582580, 0.597160, 0.584560, 0.588940, 0.595340, 0.624200, 0.607160, 0.603700, 0.596860, 0.596460, 0.612720, 0.598040, 0.624500, 0.625100, 0.623680, 0.628700, 0.633040, 0.632960, 0.642060, 0.655900, 0.645940, 0.648240, 0.644400, 0.670460, 0.682600, 0.671260, 0.669100, 0.684620, 0.694020, 0.702720, 0.702240, 0.462280, 0.462720, 0.451740, 0.452080, 0.474340, 0.481060, 0.471100, 0.430300, 0.468400, 0.795020, 0.806380, 0.799500, 0.810940, 0.466020, 0.461600, 0.446280, 0.444340, 0.447160, 0.452780, 0.477240, 0.445040, 0.471460, 0.727380, 0.697940, 0.682080, 0.684600, 0.683020, 0.683480, 0.687680, 0.665700, 0.667500, 0.652300, 0.655120, 0.616120, 0.627420, 0.632620, 0.632360, 0.623600, 0.617220, 0.603340, 0.621440, 0.626600, 0.602520, 0.627740, 0.597540, 0.610860, 0.613620, 0.600080, 0.591280, 0.597440, 0.576060, 0.600920, 0.606400, 0.572100, 0.588100, 0.582020, 0.590540, 0.575940, 0.588540, 0.567820, 0.588580, 0.580440, 0.568560, 0.560740, 0.560400, 0.570280, 0.572660, 0.576400, 0.581480, 0.568560, 0.572300, 0.574000, 0.590300, 0.567160, 0.579740, 0.579040, 0.604180, 0.580360, 0.567760, 0.568400, 0.578920, 0.576500, 0.595520, 0.592760, 0.581160, 0.583500, 0.600800, 0.583060, 0.602060, 0.595540, 0.592100, 0.612440, 0.606940, 0.627620, 0.615300, 0.602320, 0.639300, 0.630040, 0.619420, 0.630220, 0.628780, 0.659620, 0.624520, 0.665220, 0.679720, 0.659220, 0.660480, 0.671580, 0.705800, 0.681880, 0.687180, 0.700300, 0.726680, 0.714980, 0.720260, 0.616520, 0.602440, 0.611480, 0.627560, 0.611240, 0.610100, 0.620660, 0.619640, 0.612160, 0.633240, 0.629160, 0.623380, 0.611500, 0.628660, 0.605700, 0.615300, 0.630240, 0.797980, 0.821600, 0.774860, 0.763120, 0.788440, 0.780420, 0.759140, 0.751120, 0.764180, 0.768040, 0.728380, 0.721020, 0.724540, 0.714440, 0.747860, 0.718040, 0.742280, 0.704300, 0.705060, 0.693580, 0.686180, 0.679120, 0.679200, 0.677220, 0.683060, 0.681780, 0.674700, 0.685620, 0.673780, 0.667040, 0.680340, 0.652160, 0.660360, 0.670940, 0.655340, 0.649940, 0.650300, 0.655440, 0.637440, 0.649360, 0.649380, 0.665720, 0.639900, 0.644680, 0.647740, 0.635160, 0.629400, 0.628720, 0.650140, 0.648100, 0.653180, 0.636860, 0.628020, 0.635140, 0.654180, 0.639180, 0.625220, 0.646280, 0.674140, 0.657440, 0.643140, 0.654320, 0.649860, 0.667160, 0.662700, 0.664360, 0.674860, 0.682620, 0.658780, 0.676400, 0.673960, 0.678660, 0.668720, 0.681240, 0.169040, 0.143320, 0.153900, 0.175980, 0.165080, 0.149080, 0.153820, 0.157880, 0.164700, 0.132480, 0.149220, 0.147920, 0.152660, 0.152620, 0.158040, 0.153600, 0.166820, 0.772120, 0.780120, 0.813280, 0.799180, 0.831300, 0.829080, 0.854380, 0.848080, 0.844800, 0.868400, 0.868520, 0.883240, 0.893880, 0.913800, 0.903840, 0.888440, 0.893080, 0.852160, 0.832660, 0.856320, 0.828620, 0.812500, 0.811320, 0.818620, 0.799060, 0.789700, 0.765040, 0.771640, 0.154580, 0.143400, 0.157020, 0.148200, 0.139380, 0.163880, 0.153260, 0.164400, 0.172800, 0.165680, 0.188820, 0.178300, 0.152480, 0.156240, 0.168160, 0.160580, 0.154340, 0.665000, 0.678000, 0.676020, 0.672000, 0.673060, 0.673460, 0.677920, 0.661660, 0.662720, 0.664120, 0.648180, 0.642780, 0.642800, 0.655760, 0.646940, 0.652400, 0.641880, 0.643600, 0.624920, 0.641600, 0.661320, 0.634480, 0.639580, 0.645700, 0.636520]
    # totalSweepAngleInRad = 2*math.pi
    # src = plotRangeAndBearingDataOnImage(morseLandmarkTestWithNoise, totalSweepAngleInRad, IMAGE_WIDTH, IMAGE_HEIGHT)
    # ## [rangefinder_readings_to_image]

    # [load]
    filename = "morse_landmark_test_w_noise_200x200.png"
    src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_GRAYSCALE)
    if src is None:
        print ('Error opening image!')
        return -1
    ## [load]

    ## [edge_detection]
    # image = cv.Canny(image, threshold1, threshold2, aperture_size=3, L2gradient=False) returns <class 'numpy.ndarray'>
    dst = cv.Canny(image=src, threshold1=50, threshold2=200)
    ## [edge_detection]

    # Copy edges to the images that will display the results in BGR
    cdstP = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)  # <class 'numpy.ndarray'>

    ## [Probabilistic Line Transform]
    # Probabilistic Line Transform is an optimization of Standard Line Transform
    # For Probabilistic Line Transform, you must decrease the threshold
    rhoResolution = 0.5
    thetaResolution = np.pi/360
    standardThreshold = 5
    probabilisticThreshold = round(standardThreshold / 2.5)
    # cv.HoughLinesP (image, lines, rho, theta, threshold, minLineLength = 0(px?), maxLineGap = 0(px?))
    linesP = cv.HoughLinesP(image=dst, rho=rhoResolution, theta=thetaResolution, threshold=probabilisticThreshold, minLineLength=50, maxLineGap=10)
    ## [Probabilistic Line Transform]
    
    ## [draw_lines_p]
    print("Hough Transform - Probabilistic Hough Transform")
    if linesP is not None:
        for i in range(0, len(linesP)):
            # linesP[i][0] = [x1 y1 x2 y2]; e.g. [ 60  41 156  42]
            l = linesP[i][0]
            cv.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,255,0), 1, cv.LINE_AA)
            print(l)
    ## [draw_lines_p]

    # TODO: start here




    # %% find and join line segments that are segments of the same line
    # % 2. check matrix for duplicate line segments and adjust matrix to integrate
    # % them
    #         % if the segments share x-values, find the y-values that give you
    #         % the longest line
    #         % elseif the segments share y-values, find the x-values that give you
    #         % the longest line
    #     % save current row to linesMatrix2 if row is not 
        
    # % find points of intersection
    # % convert all start and end coordinates of found lines from image plane to Cartesian plane
    # % for each line, check if there are intersections
    # % if xi or yi are Inf or NaN, there is no intersection
    #         % remove invalid intersections
    #         % remove intersections that do not fall within bounds of Cartesian plane [+-134.5,+-134.5]
    
    ## [imshow]
    cv.imshow("Source", src)
    # cv.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv.imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP)
    ## [imshow]
    
    # replace return statement with this
    # publishLandmarkPositions(positions)

    ## [Wait and Exit]
    cv.waitKey()
    return 0
    ## [Wait and Exit]

def publishLandmarkPositions(positions):
    for i in range(len(positions)):
        point = Point()
        point.x = positions[i].x
        point.y = positions[i].y
        point.z = float('inf')
        landmarkPositionPublisher.publish(point)

def ROSNode():
	rospy.init_node('HoughLocalization', anonymous=False) 
	rospy.Subscriber('/RangefinderCubeBot/scan_data', LaserScan, HoughLocalization)	
	rospy.spin()	#  go into an infinite loop until it receives a shutdown signal (Ctrl+C)


if __name__ == '__main__':
    try:
        # ROSNode()
        HoughLocalization()
    except rospy.ROSInterruptException:
        pass