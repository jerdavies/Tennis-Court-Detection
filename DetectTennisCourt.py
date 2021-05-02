"""
Detect tennis courts from aerial satellite images using a trained HAAR 
classification model.

XML file with HAAR model parameters derived from the following excellent tool:
- https://amin-ahmadi.com/cascade-trainer-gui/

Base code sourced from:
- https://www.geeksforgeeks.org/detect-an-object-with-opencv-python/
"""

import cv2
from matplotlib import pyplot as plt

# File paths
test_img_path = "images/test_true_p/test1_p.jpg"
classifier_path = 'classifier/cascade3.2.xml'

# Opening image
img = cv2.imread(test_img_path)

# OpenCV opens images as BRG, converts to RGB, then to greyscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Return list of detected tennis courts in the input image
# Use minSize to ensure capture of only tennis courts and not small dots
tennis_data = cv2.CascadeClassifier(classifier_path)

found = tennis_data.detectMultiScale(img_gray, minSize=(100, 100))

# If 1+ tennis court is found, draw a rectangle around it. Else do nothing
amount_found = len(found)
print(found)

if amount_found != 0:

    # Handle images with multiple tennis courts
    for (x, y, width, height) in found:
        # We draw a green rectangle around each identified court
        cv2.rectangle(img_rgb, (x, y),
                      (x + height, y + width),
                      (0, 255, 0), 5)

# Create the environment of the picture and render it
plt.subplot(1, 1, 1)
plt.imshow(img_rgb)
plt.show()
