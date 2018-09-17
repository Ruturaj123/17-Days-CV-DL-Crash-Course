# Import the Dependencies
from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)
 
# Construct the Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to the input image")
ap.add_argument("-w", "--width", type=float, required=True,
	help="width of the left-most object in the image (in inches)")
args = vars(ap.parse_args())

# Load the image, apply grayscale and blur it
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7, 7), 0)

# Perform edge detection, erosion and dilation to close the gaps
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations = 1)
edged = cv2.erode(edged, None, iterations = 1)

# Find the contours
cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if imutils.is_cv2() else cnts[1]

# Sort the contours from left to right and initialize pixelsPerMetric variable
(cnts, _) = contours.sort_contours(cnts)
pixelsPerMetric = None

for c in cnts:
	# Ignore small contours
	if cv2.contourArea(c) < 100:
		continue

	# Compute the bounding box
	original = image.copy()
	bounding_box = cv2.minAreaRect(c)
	bounding_box = cv2.cv.BoxPoints(bounding_box) if imutils.is_cv2() else cv2.boxPoints(bounding_box)
	bounding_box = np.array(bounding_box, dtype="int")

	# Order the points in top-left, top-right, bottom-right and bottom-left order
	bounding_box = perspective.order_points(bounding_box)
	cv2.drawContours(original, [bounding_box.astype("int")], -1, (0, 255, 0), 2)

	for (x, y) in bounding_box:
		cv2.circle(original, (int(x), int(y)), 5, (0, 0, 255), -1)

	# Compute the midpoints
	(tl, tr, br, bl) = bounding_box
	(tltrX, tltrY) = midpoint(tl, tr)
	(blbrX, blbrY) = midpoint(bl, br)
	(tlblX, tlblY) = midpoint(tl, bl)
	(trbrX, trbrY) = midpoint(tr, br)

	# Draw the midpoints and the lines between them
	cv2.circle(original, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
	cv2.circle(original, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
	cv2.circle(original, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
	cv2.circle(original, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

	cv2.line(original, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)), (255, 0, 255), 2)
	cv2.line(original, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

	# Calculate the euclidean distance between the midpts
	dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
	dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

	# Compute pixelsPerMetric
	if pixelsPerMetric is None:
		pixelsPerMetric = dB / args["width"]


	# Calculate size of objects
	sizeA = dA / pixelsPerMetric
	sizeB = dB / pixelsPerMetric

	# Draw the objects
	cv2.putText(original, "{:.1f}in".format(sizeA), (int(tltrX - 15), int(tltrY - 10)), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
	cv2.putText(original, "{:.1f}in".format(sizeB), (int(trbrX + 10), int(trbrY)), 
		cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

	# Display the output
	cv2.imshow("Image", original)
	cv2.waitKey(0)
