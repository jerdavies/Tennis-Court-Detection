"""
!! WORK IN PROGRESS !!

Detect tennis courts from aerial satellite images using a trained HAAR or LBP
classification model.

XML file with model parameters derived from the following excellent tool:
- https://amin-ahmadi.com/cascade-trainer-gui/6
Some cv2 code sourced from:
- https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
"""

from markup import label_court
from transformations import rotate_image

import cv2
from matplotlib import pyplot as plt

# Constants
MIN_SIZE_DETECTED = 100  # Objects with height or width smaller than this are ignored
MIN_NEIGHBOURS = 5       # Num neighbors each candidate rect should have to retain
SCALE_FACTOR = 1.1       # Greater scale reduces propensity for positives. Must be > 1
ROTATE_COUNTER = 0        # degrees to rotate image counterclockwise

# File paths (Input variables)
test_img_path = "images/test_true_n/test5_n.jpg"
classifier_path = 'classifier/cascade7.xml'

# Open image as BRG; convert to RGB and grayscale
img = cv2.imread(test_img_path)
img_rotate = rotate_image(img, ROTATE_COUNTER)
img_gray = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img_rotate, cv2.COLOR_BGR2RGB)

# Return list of detected tennis courts in the input image
# Use minSize to ensure capture of (hopefully) only tennis courts and not small dots
tennis_data = cv2.CascadeClassifier(classifier_path)


found = tennis_data.detectMultiScale(
    img_gray, minSize=(MIN_SIZE_DETECTED,
                       MIN_SIZE_DETECTED), minNeighbors=MIN_NEIGHBOURS,
    scaleFactor=SCALE_FACTOR)

# If 1+ tennis court is found, draw a rectangle around it and label it. Else do nothing
amount_found = len(found)
print(found)

if amount_found != 0:
    for (x, y, width, height) in found:
        label_court(img_rgb, x, y, width, height)

# Create the environment of the picture and render it
plt.subplot(1, 1, 1)
plt.gcf().canvas.set_window_title(test_img_path)
plt.imshow(img_rgb)
plt.show()
