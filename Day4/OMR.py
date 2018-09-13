#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 12 18:36:17 2018

@author: ruturaj
"""

# Import the dependencies
from imutils.perspective import four_point_transform
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
 
# construct the Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
args = vars(ap.parse_args())
 
#Correct answer key
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

#load the image, convert to grayscale, apply blur, detect the edges 
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurImage = cv2.GaussianBlur(gray, (5, 5), 0)
edgedImage = cv2.Canny(blurImage, 75, 200)

#Find contours
cnts = cv2.findContours(edgedImage.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
documentContour = 0

if len(cnts) > 0:
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    
    for c in cnts:
        #Approximate the contours
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*perimeter, True)
        
        #If approximate contour has 4pts, we have found the screen
        if len(approx) == 4:
            documentContour = approx
            break

#Apply four point transform
sheet = four_point_transform(image, documentContour.reshape(4, 2))
transformed = four_point_transform(gray, documentContour.reshape(4, 2))

#Applying OTSU's thresholding method
threshold = cv2.threshold(transformed, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

#Finding contours in threshold image
cnts = cv2.findContours(threshold.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]
questionContours = []

for c in cnts:
    #Compute the bounding box and use it to find the aspect ratio
    (x, y, w, h) = cv2.boundingRect(c)
    aspectRatio = w/float(h)
    
    #Check if contour satifies the following conditon to be categorized as a question
    if w >= 20 and h >= 20 and aspectRatio >= 0.9 and aspectRatio <= 1.1:
        questionContours.append(c)
        
questionContours = contours.sort_contours(questionContours, method="top-to-bottom")[0]
correctAns = 0

#Loop over the question in batches of 5
for (q, i) in enumerate(np.arange(0, len(questionContours), 5)):
    # Sort contours from left to right
    cnts = contours.sort_contours(questionContours[i:i+5])[0]
    marked = None
    
    #Iterate over the sorted contours
    for (j, c) in enumerate(cnts):
        # Create a mask that reveals only the current mark
        mask = np.zeros(threshold.shape, dtype="uint8")
        cv2.drawContours(mask, [c], -1, 255, -1)
        
        # Apply the mask to threshold and count the number of non-zero pixels
        mask = cv2.bitwise_and(threshold, threshold, mask=mask)
        totalNonZero = cv2.countNonZero(mask)
        
        # If current mask has greater non-zero pixels then we are examining the marked ans
        if marked is None or totalNonZero > marked[0]:
            marked = (totalNonZero, j)
            
    #Initialize the index of correct ans
    color = (0, 0, 255)
    key = ANSWER_KEY[q]
    
    #Check if marked ans is correct
    if key == marked[1]:
        color = (0, 255, 0)
        correctAns += 1
        
    #Indicate the correct ans on the test
    cv2.drawContours(sheet, [cnts[key]], -1, color, 3)

#Calculate the score
score = (correctAns / 5.0)*100
cv2.putText(sheet, "{:.2f}%".format(score), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, (255, 0, 0), 2)
cv2.imshow("Original", image)
cv2.imshow("Test Score", sheet)
cv2.waitKey(0) 
