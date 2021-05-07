"""
Preprocess image folders to be used in training the classification model
- Resize each image to ensure all images have the same size
- Convert colored images to greyscale to remove excess noise
- Produce duplicate images of each img with small rotations to increase the # of training images
"""

from transformations import rotate_image, resize_to_target

import cv2
import os

from os import listdir, makedirs
from os.path import isfile, join

# Constants
TARGET_WIDTH = 60
TARGET_HEIGHT = 114
THETAS = [-3, -1, 0, 1, 3, 177, 179, 180, 181, 183]

# Set directory paths
path = 'images/n_colour'  # Source Folder
dstpath = 'images/n'  # Destination Folder

# create destination directory
try:
    makedirs(dstpath)
except:
    print("Directory already exist, images will be written in same folder")

# Create a list of all files in path
files = list(filter(lambda f: isfile(join(path, f)), listdir(path)))

# Create a new folder with images that are resized and converted to B+W
for image in files:
    try:
        img = cv2.imread(os.path.join(path, image))

        # resize image to target dimensions and convert to grayscale
        resized = resize_to_target(img, TARGET_WIDTH, TARGET_HEIGHT)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

        # for each angle in theta, produce a rotated image and write it to file
        for t in THETAS:
            image_file_name = str(t) + image
            gray_rotated = rotate_image(gray, t)

            dstPath = join(dstpath, image_file_name)
            cv2.imwrite(dstPath, gray_rotated)
    except:
        print("{} is not converted".format(image))
