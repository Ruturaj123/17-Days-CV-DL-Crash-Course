#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 19:49:01 2018

@author: ruturaj
"""

# i=Import the Dependencies
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

# Construct the Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())

# Define the upper and lower boundaries of the object in HSV
objectUpper = (29, 86, 6)
objectLower = (64, 255, 255)
tail = deque(maxlen=args["buffer"])

if not args.get("video", False):
    vs = VideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args["video"])
    
time.sleep(2.0)

# Start capturing
while True:
    #Grab the frame
    frame = vs.read()
    
    frame = frame[1] if args.get("video", False) else frame
    
    if frame is None:
        break
    
    # Process the frame
    frame = imutils.resize(frame, width=600)
    blurredFrame = cv2.GaussianBlur(frame, (11, 11), 0)
    hsvFrame = cv2.cvtColor(blurredFrame, cv2.COLOR_BGR2HSV)
    
    #Create a mask for the object and perform erosions and dilations to remove small blobs
    mask = cv2.inRange(hsvFrame, objectLower, objectUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    
    #Find contours
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    center = None
    
    if len(cnts) > 0:
        # Find largest contour and then find the center
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        
        #Proceed only if radius meets a minimum size
        if radius > 10:
            # Draw the circle and centroid and update the tail
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
        
    tail.appendleft(center)
        
    #Loop over the set of tracked points
    for i in range(1, len(tail)):
        if tail[i-1] is None or tail[i] is None:
            continue
            
        #Compute the thickness and draw the lines
        thickness = int(np.sqrt(args["buffer"]/float(i+1))*2.5)
        cv2.line(frame, tail[i-1], tail[i], (0, 0, 255), thickness)
        
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

if not args.get("video", False):
    vs.stop()
else:
    vs.release()

cv2.destroyAllWindows()    
