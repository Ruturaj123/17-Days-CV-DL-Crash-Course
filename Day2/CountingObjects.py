#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 15:45:47 2018

@author: ruturaj
"""

# Import the dependencies
import argparse
import imutils
import cv2
 
# Construct the Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# Load the image and convert it to grayscale
image = cv2.imread(args["image"])
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Applying edge detection using Canny
edge_image = cv2.Canny(gray_image, 30, 150)

#Threshold the image
threshold = cv2.threshold(gray_image, 250, 255, cv2.THRESH_BINARY_INV)[1]

#Find contours of the threshold image
contours = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if imutils.is_cv2() else contours[1]
output_image = image.copy()

#Loop over the contours and draw each of them on the output image one by one
for c in contours:
    cv2.drawContours(output_image, [c], -1, (255,165,0), 3)
    cv2.imshow("Output", output_image)
    cv2.waitKey(0)

#Draw the total number of objects detected
message = "There are {} objects".format(len(contours))
cv2.putText(output_image, message, (10, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
cv2.imshow("Output", output_image)
cv2.waitKey(0)

#Apply erosion to reduce the size of foreground objects
mask = threshold.copy()
mask = cv2.erode(mask, None, iterations=5)
cv2.imshow("Eroded", mask)
cv2.waitKey(0)

#Apply dilation to enlarge the foreground objects
mask = threshold.copy()
mask = cv2.dilate(mask, None, iterations=5)
cv2.imshow("Dilated", mask)
cv2.waitKey(0)

#Bitwise AND  for masking
mask = threshold.copy()
output = cv2.bitwise_and(image, image, mask=mask)
cv2.imshow("Bitwise AND", output)
cv2.waitKey(0)