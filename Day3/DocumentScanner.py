#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 09:13:05 2018

@author: ruturaj
"""

# Import the Dependencies
from transform import four_point_transform
from skimage.filters import threshold_local
import argparse
import cv2
import imutils
 
# Construct the Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True,
	help = "Path to the image to be scanned")
args = vars(ap.parse_args())

# Load the image and calulate the ratio of old height to new height
image = cv2.imread(args["image"])
ratio = image.shape[0]/500.0
original = image.copy()
image = imutils.resize(image, height=500)

# Convert image to grayscale, blur it and find the edges
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(gray, 75, 200)

# Show the original image and the edge detected image
cv2.imshow("Gray Image", image)
cv2.imshow("Edged Image", edges)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Find contours in the edged image
contours = cv2.findContours(edges.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = contours[0] if imutils.is_cv2() else contours[1]
contours = sorted(contours, key = cv2.contourArea, reverse=True)[:5]

#Iterate over the contours
for cnt in contours:
    #Approximate the contours
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02*perimeter, True)
    
    #If approximate contour has 4pts, we have found the screen
    if len(approx) == 4:
        screenContour = approx
        break
    
#Display the contour
cv2.drawContours(image, [screenContour], -1, (0, 255, 0), 2)
cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()    

transformed = four_point_transform(original, screenContour.reshape(4,2)*ratio)

#Convert transformed image to grayscale and apply threshold
transformed = cv2.cvtColor(transformed, cv2.COLOR_BGR2GRAY)
threshold = threshold_local(transformed, 11, offset=10, method="gaussian")
transformed = (transformed > threshold).astype("uint8")*255

cv2.imshow("Original", imutils.resize(original, height = 650))
cv2.imshow("Scanned", imutils.resize(transformed, height = 650))
cv2.waitKey(0)
