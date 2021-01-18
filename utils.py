import numpy as np
import math

LEFT_SHOULDER = 11
LEFT_ELBOW = 13
LEFT_WRIST = 15
LEFT_HIP = 23

def distance(a, b):
    point1 = np.array([a.x, a.y])
    point2 = np.array([b.x, b.y])
    return np.linalg.norm(point1 - point2)

def angle(a,b,c):
    # law of cosines to angle in b
    side_a = distance(b, c)
    side_b = distance(a, c)
    side_c = distance(a, b)
    numerator = np.square(side_a)+np.square(side_c)-np.square(side_b)
    cosine_b = numerator/(2*side_a*side_c)
    angle_b = np.arccos(cosine_b)
    return np.degrees(angle_b)


def angle_2p_3d(a, b, c):       

    v1 = np.array([ a[0] - b[0], a[1] - b[1], a[2] - b[2] ])
    v2 = np.array([ c[0] - b[0], c[1] - b[1], c[2] - b[2] ])

    v1mag = np.sqrt([ v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2] ])
    v1norm = np.array([ v1[0] / v1mag, v1[1] / v1mag, v1[2] / v1mag ])

    v2mag = np.sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2])
    v2norm = np.array([ v2[0] / v2mag, v2[1] / v2mag, v2[2] / v2mag ])
    res = v1norm[0] * v2norm[0] + v1norm[1] * v2norm[1] + v1norm[2] * v2norm[2]
    angle_rad = np.arccos(res)

    return math.degrees(angle_rad)


def angle_3d(a,b,c):
    a = [a.x,a.y,a.z]
    b = [b.x,b.y,b.z]
    c = [c.x,c.y,c.z]
    return angle_2p_3d(a,b,c)