#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep  9 08:31:57 2018

@author: ruturaj
"""

# Import the dependencies
import numpy as np
import argparse
import cv2
 
# Construct the argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-p", "--prototxt", required=True,
	help="path to Caffe 'deploy' prototxt file")
ap.add_argument("-m", "--model", required=True,
	help="path to Caffe pre-trained model")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

#Loading the model
print("Loading...")
net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

#Read Image and create a blob
image = cv2.imread(args["image"])
(height, width) = image.shape[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), 
                             (104.0, 177.0, 123.0))

#Forward propagate the blob through the network and get the predictions
net.setInput(blob)
detections = net.forward()

#Iterate over the detections
for i in range(0, detections.shape[2]):
    #Get the confidence 
    confidence= detections[0, 0, i, 2]
    
    #Check the confidence level
    if confidence > args['confidence']:
        #Get the bounding box coordinates
        bounding_box = detections[0, 0, i, 3:7]*np.array([width, height, width, height])
        (startX, startY, endX, endY) = bounding_box.astype('int')
        
        #Draw the bounding box
        confidence_level_text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY > 10 else startY + 10
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(image, confidence_level_text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        
#Display the output
cv2.imshow("Output", image)
cv2.waitKey(0)