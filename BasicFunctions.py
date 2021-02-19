# Course followed: https://youtu.be/oXlwWbU8l2o

import cv2 as cv  # self-reminder to first open VScode from anaconda prompt

# method that takes in path to image and returns image as a matrix of pixels
img = cv.imread('images/146.jpg')

# display image as new window
cv.imshow('Testphoto', img)

# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# Blur. Note: increase blur by increasing kernel size (which must be odd num)
blur = cv.GaussianBlur(img, (3, 3), cv.BORDER_DEFAULT)
cv.imshow('Blur', blur)

# Edge cascade - find edges present in image using Canny edge detector
# can reduce number of edges when passing in blurred image
canny = cv.Canny(img, 125, 175)
cv.imshow('Canny Edges', canny)

# Dilating the image (making the shapes in the image larger essentially)
dilated = cv.dilate(canny, (3, 3), iterations=3)
cv.imshow('Dilated', dilated)

# Eroding
eroded = cv.erode(dilated, (3, 3), iterations=3)
cv.imshow("Eroded", eroded)

# Resizing
resized = cv.resize(img, (500, 500))
cv.imshow("Resized", resized)

# wait indefinitely until a key is pressed
cv.waitKey(0)
