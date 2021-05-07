"""
!! WORK IN PROGRESS - still not functional !!

Detect tennis courts from aerial satellite images using geometric 
approach with lines and contours (continuous lines).

Portions of code derived from:
- Mask Loop: https://stackoverflow.com/questions/55496402/detecting-tennis-court-lines-intercepts
- Contour filtering: https://stackoverflow.com/questions/61166180/detect-rectangles-in-opencv-4-2-0-using-python-3-7
"""

import numpy as np
import cv2

# Constants
MIN_RGB = 0           # darkest RGB pixels to keep
MIN_THRESHOLD = 100   # below this RGB value will be converted to black
KERNEL_SIZE = 9

# read image
img = cv2.imread('images/149.jpg')

# define the list of colour boundaries (keeping only light to white colours)
boundaries = [([MIN_RGB, MIN_RGB, MIN_RGB], [255, 255, 255])]

# loop over the boundaries and filter out (mask) dark pixels
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

# convert image to grayscale and apply thresholding (pixel becomes only B or W)
gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, MIN_THRESHOLD, 255, cv2.THRESH_BINARY)
# thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                              cv2.THRESH_BINARY, 11, 2)

# detect edges
blur = cv2.GaussianBlur(thresh, (KERNEL_SIZE, KERNEL_SIZE), 1)
canny = cv2.Canny(blur, 10, 50)

# find all contours (curves joining continuous pts, having same color or intensity)
contours = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

# only draw contours that meet filter criteria (rectangles of minimum size)
count_contours = 0

for cnt in contours:
    epsilon = 0.05*cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)
    if cv2.contourArea(cnt) > 1000 and len(approx) == 4:
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(approx)
        cv2.putText(img, "tennis court", (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        count_contours += 1

# display original image with contours drawn
print("# of contours is :" + str(count_contours))
cv2.imshow('Contours', img)
cv2.waitKey(0)
