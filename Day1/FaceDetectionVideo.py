#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 09:52:46 2018

@author: ruturaj
"""

# Import the dependencies
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import cv2
 
# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# Load the model
print("Loading...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#Initialze the video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

#Loop over the video frames
while True:
    #Resize the frame 
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    
    #Convert frame to blob
    (height, width) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    
    #Forward propagate the blob through the network and get the detections
    net.setInput(blob)
    detections = net.forward()
    
    #Iterate over the detections
    for i in range(0, detections.shape[2]):
		#Get the confidence
        confidence = detections[0, 0, i, 2]
 
		#Check the confidence level
        if confidence < args["confidence"]:
            continue
 
		#Get the bounding box coordinates
        box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
        (startX, startY, endX, endY) = box.astype("int")
 
		#Draw the bounding box
        confidence_level_text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, confidence_level_text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord("q"):
        break
    
cv2.destroyAllWindows()
vs.stop()