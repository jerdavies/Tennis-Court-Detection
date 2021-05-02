"""
Find the intersection points of lines.

 Code from here:
- auto_canny: https://www.pyimagesearch.com/2015/04/06/zero-parameter-automatic-canny-edge-detection-with-python-and-opencv/
"""

import numpy as np
import cv2


def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)
    # return the edged image
    return edged


# read image
img = cv2.imread('images/sudoku.png')
img = cv2.imread('images/149.jpg')

# define the list of boundaries (keeping only light to white colours)
boundaries = [([90, 90, 90], [255, 255, 255])]

# loop over the boundaries and filter out (mask) dark pixels
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply the mask
    mask = cv2.inRange(img, lower, upper)
    output = cv2.bitwise_and(img, img, mask=mask)

# show the images: original vs. only light kept
#cv2.imshow("images", np.hstack([img, output]))
# cv2.waitKey(0)

gray = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (3, 3), 0)
low_threshold, high_threshold = 100, 500
#edges = cv2.Canny(gray, low_threshold, high_threshold)

edges = auto_canny(blurred)

cv2.imshow('edges', edges)
cv2.waitKey(0)

rho = 3  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 200  # minimum number of votes (intersections in Hough grid cell)
# min_line_length = 10  # minimum number of pixels making up a line
# max_line_gap = 5  # maximum gap in pixels between connectable line segments
line_image = np.copy(img) * 0  # creating a blank to draw lines on

lines = cv2.HoughLines(edges, rho, theta, threshold)

print(len(lines))
for line in lines:
    rho, theta = line[0]
    if (rho > 200):
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho

        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))

        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)

cv2.imshow('houghlines', img)
cv2.waitKey(0)
