import cv2 as cv  # self-reminder to first open VScode from anaconda prompt
import numpy as np

# method that takes in path to image and returns image as a matrix of pixels
img = cv.imread('images/146.jpg')

# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

# Blur. Note: increase blur by increasing kernel size (which must be odd num)
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)

# CANNY EDGE DETECTION

# Edge cascade - find edges present in image using Canny edge detector
# can reduce number of edges when passing in blurred image
canny = cv.Canny(gray, 100, 175)
cv.imshow('Canny Edges', canny)

# LAPLACIAN METHOD - appears like a pencil shading of the image
lap = cv.Laplacian(gray, cv.CV_64F)
lap = np.uint8(np.absolute(lap))
#cv.imshow('Laplacian', lap)

#

# wait indefinitely until a key is pressed
cv.waitKey(0)
