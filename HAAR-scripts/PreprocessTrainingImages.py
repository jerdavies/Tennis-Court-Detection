"""
Preprocess image folders to be used in training the classification model
- Resize each image to ensure all images have the same size
- Convert colored images to greyscale to remove excess noise
- Produce duplicate images of each img with small rotations to increase the # of training images
"""

import cv2
import os
import glob
import numpy as np

from os import listdir, makedirs
from os.path import isfile, join

# Constants
TARGET_WIDTH = 60
TARGET_HEIGHT = 114
THETAS = [-3, -1, 0, 1, 3, 177, 179, 180, 181, 183]

# Set directory paths
path = 'images/n_colour'  # Source Folder
dstpath = 'images/n'  # Destination Folder

# Helper functions


def rotate_image(image, angle):
    """
    Rotate image counterclockwise by angle (degrees).
    Finds the center of image, calculates the transformation matrix, and applies to the image
    """
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(
        (image_center[0], image_center[1]), angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# create destination directory
try:
    makedirs(dstpath)
except:
    print("Directory already exist, images will be written in same folder")

# Scale and convert to B+W
files = list(filter(lambda f: isfile(join(path, f)), listdir(path)))
for image in files:
    try:
        img = cv2.imread(os.path.join(path, image))

        # scale width to TARGET_WIDTH and preserve aspect ratio
        width = int(img.shape[1])
        height = int(img.shape[0])
        dim = (TARGET_WIDTH, TARGET_HEIGHT)
        resized = cv2.resize(img, dim)

        # crop image height to TARGET_HEIGHT
        # cropped = resized[0:TARGET_HEIGHT, 0:TARGET_WIDTH]

        # convert cropped image to greyscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # for each angle in theta, produce a rotated image and write it to file
        for t in THETAS:
            image_file_name = str(t) + image
            gray_rotated = rotate_image(gray, t)

            dstPath = join(dstpath, image_file_name)
            cv2.imwrite(dstPath, gray_rotated)
    except:
        print("{} is not converted".format(image))

for fil in glob.glob("*.jpg"):
    try:
        image = cv2.imread(fil)
        gray_image = cv2.cvtColor(os.path.join(
            path, image), cv2.COLOR_BGR2GRAY)  # convert to greyscale
        cv2.imwrite(os.path.join(dstpath, fil), gray_image)
    except:
        print('{} is not converted')
