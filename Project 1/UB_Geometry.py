import numpy as np
from typing import List, Tuple
import cv2

from cv2 import cvtColor, COLOR_BGR2GRAY, TERM_CRITERIA_EPS, TERM_CRITERIA_MAX_ITER, \
    findChessboardCorners, cornerSubPix, drawChessboardCorners

'''
Please do Not change or add any imports. 
'''

#task1

def findRot_xyz2XYZ(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles along x, y and z axis respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from xyz to XYZ.

    '''
    rot_xyz2XYZ = np.eye(3).astype(float)

    # Your implementation
    
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    r1 = [np.cos(alpha),-1*np.sin(alpha),0]
    r2 = [np.sin(alpha),np.cos(alpha),0]
    r3 = [0,0,1]
    zalpha = np.array([r1,r2,r3])

    r1 = [1,0,0]
    r2 = [0,np.cos(beta),-1*np.sin(beta)]
    r3 = [0,np.sin(beta),np.cos(beta)]
    xbeta = np.array([r1,r2,r3])

    r1 = [np.cos(gamma),-1*np.sin(gamma),0]
    r2 = [np.sin(gamma),np.cos(gamma),0]
    r3 = [0,0,1]
    zgamma = np.array([r1,r2,r3])

    rot_xyz2XYZ = (zalpha.dot(xbeta)).dot(zgamma)

    return rot_xyz2XYZ

def findRot_XYZ2xyz(alpha: float, beta: float, gamma: float) -> np.ndarray:
    '''
    Args:
        alpha, beta, gamma: They are the rotation angles of the 3 step respectly.
            Note that they are angles, not radians.
    Return:
        A 3x3 numpy array represents the rotation matrix from XYZ to xyz.

    '''
    rot_XYZ2xyz = np.eye(3).astype(float)

    # Your implementation

    alpha = -1*np.deg2rad(alpha)
    beta = -1*np.deg2rad(beta)
    gamma = -1*np.deg2rad(gamma)

    r1 = [np.cos(alpha),-1*np.sin(alpha),0]
    r2 = [np.sin(alpha),np.cos(alpha),0]
    r3 = [0,0,1]
    zalpha = np.array([r1,r2,r3])

    r1 = [1,0,0]
    r2 = [0,np.cos(beta),-1*np.sin(beta)]
    r3 = [0,np.sin(beta),np.cos(beta)]
    xbeta = np.array([r1,r2,r3])

    r1 = [np.cos(gamma),-1*np.sin(gamma),0]
    r2 = [np.sin(gamma),np.cos(gamma),0]
    r3 = [0,0,1]
    zgamma = np.array([r1,r2,r3])

    rot_XYZ2xyz = (zalpha.dot(xbeta)).dot(zgamma)
    return rot_XYZ2xyz


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above "findRot_xyz2XYZ()" and "findRot_XYZ2xyz()" functions are the only 2 function that will be called in task1.py.
"""

# Your functions for task1


#--------------------------------------------------------------------------------------------------------------
# task2:

def find_corner_img_coord(image: np.ndarray) -> np.ndarray:
    '''
    Args: 
        image: Input image of size MxNx3. M is the height of the image. N is the width of the image. 3 is the channel of the image.
    Return:
        A numpy array of size 32x2 that represents the 32 checkerboard corners' pixel coordinates. 
        The pixel coordinate is defined such that the of top-left corner is (0, 0) and the bottom-right corner of the image is (N, M). 
    '''
    img_coord = np.zeros([32, 2], dtype=float)

    # Your implementation
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (4,9))

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corners, (11,11),(-1,-1), (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
        image = cv2.drawChessboardCorners(image, (4,9), corners2, ret)
    #cv2_imshow(img)

    #Removing the middle box of corners on Z axis
    l = corners2[:16].tolist() + corners2[20:].tolist()
    corners2 = np.array(l)
    count = 0
    for i in corners2:
        img_coord[count] = i
        count += 1
    return img_coord


def find_corner_world_coord(img_coord: np.ndarray) -> np.ndarray:
    '''
    You can output the world coord manually or through some algorithms you design. Your output should be the same order with img_coord.
    Args: 
        img_coord: The image coordinate of the corners. Note that you do not required to use this as input, 
        as long as your output is in the same order with img_coord.
    Return:
        A numpy array of size 32x3 that represents the 32 checkerboard corners' pixel coordinates. 
        The world coordinate or each point should be in form of (x, y, z). 
        The axis of the world coordinate system are given in the image. The output results should be in milimeters.
    '''
    world_coord = np.zeros([32, 3], dtype=float)  

    # Your implementation

    #Around X and Z axis
    world_coord[0] = [40,0,40]
    world_coord[1] = [40,0,30]
    world_coord[2] = [40,0,20]
    world_coord[3] = [40,0,10]
    world_coord[4] = [30,0,40]
    world_coord[5] = [30,0,30]
    world_coord[6] = [30,0,20]
    world_coord[7] = [30,0,10]
    world_coord[8] = [20,0,40]
    world_coord[9] = [20,0,30]
    world_coord[10] = [20,0,20]
    world_coord[11] = [20,0,10]
    world_coord[12] = [10,0,40]
    world_coord[13] = [10,0,30]
    world_coord[14] = [10,0,20]
    world_coord[15] = [10,0,10]

    #Around Y and Z axis
    world_coord[16] = [0,10,40]
    world_coord[17] = [0,10,30]
    world_coord[18] = [0,10,20]
    world_coord[19] = [0,10,10]
    world_coord[20] = [0,20,40]
    world_coord[21] = [0,20,30]
    world_coord[22] = [0,20,20]
    world_coord[23] = [0,20,10]
    world_coord[24] = [0,30,40]
    world_coord[25] = [0,30,30]
    world_coord[26] = [0,30,20]
    world_coord[27] = [0,30,10]
    world_coord[28] = [0,40,40]
    world_coord[29] = [0,40,30]
    world_coord[30] = [0,40,20]
    world_coord[31] = [0,40,10]

    return world_coord


def find_intrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[float, float, float, float]:
    '''
    Use the image coordinates and world coordinates of the 32 point to calculate the intrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        fx, fy: Focal length. 
        (cx, cy): Principal point of the camera (in pixel coordinate).
    '''

    fx: float = 0
    fy: float = 0
    cx: float = 0
    cy: float = 0

    # Your implementation
    M = estimate_M(img_coord,world_coord)
    fx,fy,cx,cy = calculate(M)
    return fx, fy, cx, cy


def find_extrinsic(img_coord: np.ndarray, world_coord: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    '''
    Use the image coordinates, world coordinates of the 32 point and the intrinsic parameters to calculate the extrinsic parameters.
    Args: 
        img_coord: The image coordinate of the 32 corners. This is a 32x2 numpy array.
        world_coord: The world coordinate of the 32 corners. This is a 32x3 numpy array.
    Returns:
        R: The rotation matrix of the extrinsic parameters. It is a 3x3 numpy array.
        T: The translation matrix of the extrinsic parameters. It is a 1-dimensional numpy array with length of 3.
    '''

    R = np.eye(3).astype(float)
    T = np.zeros(3, dtype=float)

    # Your implementation
    M = estimate_M(img_coord,world_coord)
    fx,fy,cx,cy = calculate(M)
    M_in = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
    M_in_inv = np.linalg.inv(M_in)
    M_ex = M_in_inv.dot(M)
    #Rotation matrix estimation
    R[0][0],R[0][1],R[0][2] = M_ex[0][0],M_ex[0][1],M_ex[0][2]
    R[1][0],R[1][1],R[1][2] = M_ex[1][0],M_ex[1][1],M_ex[1][2]
    R[2][0],R[2][1],R[2][2] = M_ex[2][0],M_ex[2][1],M_ex[2][2]
    #Translation matrix estimation
    T[0] = M_ex[0][3]
    T[1] = M_ex[1][3]
    T[2] = M_ex[2][3]
    return R, T


"""
If your implementation requires implementing other functions. Please implement all the functions you design under here.
But remember the above 4 functions are the only ones that will be called in task2.py.
"""

# Your functions for task2
def calculate(M):
    m11 = M[0][0]
    m12 = M[0][1]
    m13 = M[0][2]
    m14 = M[0][3]
    m21 = M[1][0]
    m22 = M[1][1]
    m23 = M[1][2]
    m24 = M[1][3]
    m31 = M[2][0]
    m32 = M[2][1]
    m33 = M[2][2]
    m34 = M[2][3]
    m1 = np.array([[m11,m12,m13]])
    m2 = np.array([[m21,m22,m23]])
    m3 = np.array([[m31,m32,m33]])
    m4 = np.array([[m14,m24,m34]])
    m1 = np.transpose(m1)
    m2 = np.transpose(m2)
    m3 = np.transpose(m3)
    m4 = np.transpose(m4)

    cx = float((np.transpose(m1)).dot(m3))
    cy = float((np.transpose(m2)).dot(m3))
    fx = float(np.sqrt(np.transpose(m1).dot(m1)-cx*cx))
    fy = float(np.sqrt(np.transpose(m2).dot(m2)-cy*cy))
    return fx,fy,cx,cy

def estimate_M(img_coord,world_coord):
    A = np.zeros([64,12],dtype=float)
    count = 0
    for i in range(0,32):
        X_c,Y_c = img_coord[i]
        X_w,Y_w,Z_w = world_coord[i]
        A[count] = [X_w,Y_w,Z_w,1,0,0,0,0,-1*X_c*X_w,-1*X_c*Y_w,-1*X_c*Z_w,-1*X_c]
        count += 1
        A[count] = [0,0,0,0,X_w,Y_w,Z_w,1,-1*Y_c*X_w,-1*Y_c*Y_w,-1*Y_c*Z_w,-1*Y_c]
        count += 1
    U,D,V = np.linalg.svd(A)
    x = V[-1].reshape((3,4))
    l = x[-1][:3]
    norm_x = np.linalg.norm(l)
    M = x*(1/norm_x)
    return M


#---------------------------------------------------------------------------------------------------------------------