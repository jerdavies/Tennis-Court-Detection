import cv2 as cv

# method that takes in path to image and returns image as a matrix of pixels
img = cv.imread('images/146.jpg')

# display image as new window
cv.imshow('Testphoto', img)

# Convert image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow('Gray', gray)

# wait until a key is pressed
cv.waitKey(0)
