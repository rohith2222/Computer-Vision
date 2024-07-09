'''
Notes:
1. All of your implementation should be in this file. This is the ONLY .py file you need to edit & submit. 
2. Please Read the instructions and do not modify the input and output formats of function detect_faces() and cluster_faces().
3. If you want to show an image for debugging, please use show_image() function in helper.py.
4. Please do NOT save any intermediate files in your final submission.
'''


import cv2
import numpy as np
import os
import sys
import math

import face_recognition

from typing import Dict, List
from utils import show_image

'''
Please do NOT add any imports. The allowed libraries are already imported for you.
'''

def detect_faces(img: np.ndarray) -> List[List[float]]:
    """
    Args:
        img : input image is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).

    Returns:
        detection_results: a python nested list. 
            Each element is the detected bounding boxes of the faces (may be more than one faces in one image).
            The format of detected bounding boxes a python list of float with length of 4. It should be formed as 
            [topleft-x, topleft-y, box-width, box-height] in pixels.
    """
    detection_results: List[List[float]] = [] # Please make sure your output follows this data format.
    
    # Add your code here. Do not modify the return and input arguments.
    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    face_locations = face_recognition.face_locations(imgGray)
    for loc in face_locations:
        top, right, bottom, left = loc
        topleft_x, topleft_y = float(left), float(top)
        box_width = float(right-left)
        box_height = float(bottom - top)
        detection_results.append([topleft_x, topleft_y, box_width, box_height])
    return detection_results


def cluster_faces(imgs: Dict[str, np.ndarray], K: int) -> List[List[str]]:
    """
    Args:
        imgs : input images. It is a python dictionary
            The keys of the dictionary are image names (without path).
            Each value of the dictionary is an np.ndarray represent an input image of shape H x W x 3.
            H is the height of the image, W is the width of the image. 3 is the [R, G, B] channel (NOT [B, G, R]!).
        K: Number of clusters.
    Returns:
        cluster_results: a python list where each elemnts is a python list.
            Each element of the list a still a python list that represents a cluster.
            The elements of cluster list are python strings, which are image filenames (without path).
            Note that, the final filename should be from the input "imgs". Please do not change the filenames.
    """
    cluster_results: List[List[str]] = [[]] * K # Please make sure your output follows this data format.

    # Add your code here. Do not modify the return and input arguments.
    face_encodes = {}
    for name in imgs:
        img = imgs[name]
        imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        face_locations = face_recognition.face_locations(imgGray)
        boxes = []
        for top, right, bottom, left in face_locations:
            face = (top, right, bottom, left)
            boxes.append(face)
        encoding = face_recognition.face_encodings(img,boxes)
        face_encodes[name] = np.array(encoding)
    cluster_results= KMeans(face_encodes,K,iterations=10)
    return cluster_results

'''
If your implementation requires multiple functions. Please implement all the functions you design under here.
But remember the above 2 functions are the only functions that will be called by task1.py and task2.py.
'''

# Your functions. (if needed)
def KMeans(encodings,K,iterations):
    img_names = list(encodings.keys())
    #Chossing K random points as centroids
    c_points = []
    for i in range(0,len(img_names)-1,int(len(img_names)/K)+1):
        c_points.append(img_names[i])
    c_points = list(c_points)
    centroids = [encodings[i] for i in c_points]
    count = 1
    cluster_results = []
    while True:
        count+=1
        old_centroids = centroids
        cluster = []
        new_centrois = []
        for i in range(K):
            cluster.append([])
            new_centrois.append([])
        for name in encodings:
            img = encodings[name]
            #Find the distance (Euclidean distance for our purpose) between each data points in our training set with the k centroids
            d = np.sqrt(np.sum(np.square(img - centroids[0])))
            pos = 0
            for i in range(1,K):
                d1 = np.sqrt(np.sum(np.square(img - centroids[i])))
                if d1<d:
                    d = d1
                    pos = i
            #Now assign each data point to the closest centroid according to the distance found
            cluster[pos].append(name)
            new_centrois[pos].append(img)
        #Update centroid location by taking the average of the points in each cluster group
        for i in range(K):
            x = np.array(new_centrois[i])
            centroids[i] = np.mean(x,axis=0)
        #Performing K means operation for the number of iterations provided
        if count==iterations:
            cluster_results = cluster
            break
    return cluster_results
    