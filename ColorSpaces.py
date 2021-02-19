import cv2 as cv  # self-reminder to first open VScode from anaconda prompt

img = cv.imread('images/147.jpg')

# BGR to Grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# display image as new window
cv.imshow('Grey', gray)

# wait until a key is pressed
cv.waitKey(0)
