"""
Preprocess image folders to be used in training the classification model
- Resize and crop to ensure all images have the same size
- Convert colored images to greyscale to remove excess noise
"""

import cv2
import os
import glob

from os import listdir, makedirs
from os.path import isfile, join

# Constants
TARGET_WIDTH = 160
TARGET_HEIGHT = 320

# Set directory paths
path = 'images/p_colour'  # Source Folder
dstpath = 'images/p'  # Destination Folder

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
        scale_factor = TARGET_WIDTH / width
        dim = (TARGET_WIDTH, int(height * scale_factor))
        resized = cv2.resize(img, dim)

        # crop image height to TARGET_HEIGHT
        cropped = resized[0:TARGET_HEIGHT, 0:TARGET_WIDTH]

        # convert cropped image to greyscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        dstPath = join(dstpath, image)
        cv2.imwrite(dstPath, gray)
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
