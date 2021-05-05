"""
Detect tennis courts from aerial satellite images using a trained HAAR
classification model.

XML file with HAAR model parameters derived from the following excellent tool:
- https://amin-ahmadi.com/cascade-trainer-gui/

Base code sourced from:
- https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt


# File paths
test_img_path = "images/test_objects2.jpg"
classifier_path = 'classifier/cascade4.xml'


def rotate_image(image, angle):

    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(
        (image_center[0], image_center[1]), angle, 1.0)
    result = cv2.warpAffine(
        image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


# Opening image
img = cv2.imread(test_img_path)
# img = rotate_image(img, -55)
# dim = (700, 500)
# img = cv2.resize(img, dim)

# OpenCV opens images as BRG, converts to RGB and to greyscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Return list of detected tennis courts in the input image
# Use minSize to ensure capture of only tennis courts and not small dots
tennis_data = cv2.CascadeClassifier(classifier_path)

found = tennis_data.detectMultiScale(
    img_gray, minSize=(100, 100), minNeighbors=3)
# found = tennis_data.det

# If 1+ tennis court is found, draw a rectangle around it. Else do nothing
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
