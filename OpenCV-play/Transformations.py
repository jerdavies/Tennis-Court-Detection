""" Note(s):
- When working with OpenCV Python, images are stored in numpy ndarray
height = img.shape[0]
width = img.shape[1]
channels = img.shape[2]

- What is an Affine Transformation?
A transformation that can be expressed in the form of a matrix multiplication (linear transformation) followed by a vector addition (translation).
From the above, we can use an Affine Transformation to express:
--Rotations (linear transformation)
--Translations (vector addition)
--Scale operations (linear transformation)
"""

import cv2 as cv  # self-reminder to first open VScode from anaconda prompt
import numpy as np

# method that takes in path to image and returns image as a matrix of pixels
img = cv.imread('images/147.jpg')

# display image as new window
cv.imshow('Testphoto', img)


def translate(img, x, y):
    """Translate image by x (+ive is right) and y (+ive is down) pixels.
    @type img: image
    @type x, y: int
    @rtype: image"""
    transMat = np.float32([[1, 0, x], [0, 1, y]])  # transition matrix
    dimensions = (img.shape[1], img.shape[0])  # width, height
    return cv.warpAffine(img, transMat, dimensions)


# Translate
translated = translate(img, 100, 100)
cv.imshow('Translated', translated)


def rotate(img, angle, rotPoint=None):
    """Rotate image by given angle (+ive is counterclockwise).
    @type img: image
    @type angle: float
    @rytpe: image"""

    (height, width) = img.shape[:2]

    if rotPoint is None:
        rotPoint = (width//2, height//2)

    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    dimensions = (width, height)

    return cv.warpAffine(img, rotMat, dimensions)


# Rotate
rotated = rotate(img, 45)
cv.imshow('Rotated', rotated)

# Flip
flip = cv.flip(img, 0)  # 0 =vert flip; 1=hznt flip
cv.imshow('Flip', flip)

# Crop
cropped = img[200:400, 300:400]
cv.imshow('Cropped', cropped)


# wait indefinitely until a key is pressed
cv.waitKey(0)
