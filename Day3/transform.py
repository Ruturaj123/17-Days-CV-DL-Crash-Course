#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 08:56:09 2018

@author: ruturaj
"""

import cv2
import numpy as np

def order_points(points):
    box = np.zeros((4,2), dtype="float32")
    
    pts_sum = points.sum(axis=1)
    box[0] = points[np.argmin(pts_sum)]
    box[2] = points[np.argmax(pts_sum)]
    
    pts_diff = np.diff(points, axis=1)
    box[1] = points[np.argmin(pts_diff)]
    box[3] = points[np.argmax(pts_diff)]
    
    return box

def four_point_transform(image, points):
    box = order_points(points)
    (tl, tr, br, bl) = box
    
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    destination_points = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
    
    M = cv2.getPerspectiveTransform(box, destination_points)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped