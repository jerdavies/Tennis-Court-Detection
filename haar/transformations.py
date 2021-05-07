"""
Module with definitions for image transformations such as
- rotate_image
- dilate_image
"""

import cv2
import numpy as np


def resize_to_target(img, width, height):
    """
    Return resized img to given width and height dimensions
    """
    dim = (width, height)
    resized = cv2.resize(img, dim)
    return resized


def rotate_image(image, angle):
    """
    Return image rotated counterclockwise by angle (degrees).
    Finds the center of image, calculates the transformation matrix, and applies to the image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(
        (image_center[0], image_center[1]), angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result
