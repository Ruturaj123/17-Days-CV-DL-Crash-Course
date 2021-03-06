#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 22:00:23 2018

@author: ruturaj
"""

# Import the dependencies
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import numpy as np
import playsound
import argparse
import imutils
import time
import dlib
import cv2

def sound_alarm(path):
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    
    eyeAspectRatio = (A + B) / (2.0*C)
    
    return eyeAspectRatio

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
ap.add_argument("-a", "--alarm", type=str, default="", help="path alarm .WAV file")
args = vars(ap.parse_args())

EYE_AR_THRESHOLD = 0.20
EYE_AR_CONSECUTIVE_FRAMES = 50

FRAME_COUNTER = 0
ALARM_ON = False

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

(left_eye_start, left_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_eye_start, right_eye_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

vs = VideoStream(src=0).start()
fileStream = False
time.sleep(1.0)

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rectangles = detector(frame, 0)
    
    for rectangle in rectangles:
        shape = predictor(gray, rectangle)
        shape = face_utils.shape_to_np(shape)
        
        left_eye = shape[left_eye_start:left_eye_end]
        right_eye = shape[right_eye_start:right_eye_end]
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        
        EAR = (left_EAR + right_EAR)/2.0
        
        left_eye_hull = cv2.convexHull(left_eye)
        right_eye_hull = cv2.convexHull(right_eye)
        
        cv2.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)
        
        if EAR < EYE_AR_THRESHOLD:
            FRAME_COUNTER += 1

            if FRAME_COUNTER >= EYE_AR_CONSECUTIVE_FRAMES:
                if not ALARM_ON:
                    ALARM_ON = True

                    if args["alarm"] != "":
                        t = Thread(target=sound_alarm, args=(args["alarm"],))
                        t.deamon = True
                        t.start()
            
                cv2.putText(frame, "***DROWSINESS ALERT!***", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        else:
            FRAME_COUNTER = 0
            ALARM_ON = False
        cv2.putText(frame, "EAR: {:.2f}".format(EAR), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        	break

cv2.destroyAllWindows()
vs.stop()