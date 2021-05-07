"""
!! WORK IN PROGRESS !!

Detect tennis courts from aerial satellite images using a trained HAAR
classification model.

XML file with HAAR model parameters derived from the following excellent tool:
- https://amin-ahmadi.com/cascade-trainer-gui/

Base code sourced from:
- https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
"""

from transformations import rotate_image

import cv2
from matplotlib import pyplot as plt

# Constants
MIN_SIZE_DETECTED = 100  # Objects with height or width smaller than this are ignored
MIN_NEIGHBOURS = 3      # Num neighbors each candidate rect should have to retain it

# File paths (Input variables)
test_img_path = "images/test_true_p/test3_p.jpg"
classifier_path = 'classifier/cascade1.xml'

# Opening image
img = cv2.imread(test_img_path)
# img = rotate_image(img, -55)
# dim = (700, 500)
# img = cv2.resize(img, dim)

# OpenCV opens images as BRG, so convert to RGB and to greyscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Return list of detected tennis courts in the input image
# Use minSize to ensure capture of (hopefully!) only tennis courts and not small dots
tennis_data = cv2.CascadeClassifier(classifier_path)

found = tennis_data.detectMultiScale(
    img_gray, minSize=(MIN_SIZE_DETECTED, MIN_SIZE_DETECTED), minNeighbors=MIN_NEIGHBOURS)

# If 1+ tennis court is found, draw a rectangle around it and label it. Else do nothing
amount_found = len(found)
print(found)

if amount_found != 0:

    # Handle images with multiple tennis courts
    for (x, y, width, height) in found:
        # We draw a green rectangle around each identified court
        cv2.rectangle(img_rgb, (x, y),
                      (x + width, y + height),
                      (0, 255, 0), 5)
        cv2.putText(img_rgb, "tennis", (x + int((width/8)), y + int((height/3))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_rgb, "court", (x + int((width/8)), y + 30 + int((height/3))),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

# Create the environment of the picture and render it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()