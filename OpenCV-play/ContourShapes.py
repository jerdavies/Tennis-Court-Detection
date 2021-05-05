import numpy as np
import cv2

img = cv2.imread('images/149.jpg')  # read image

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

contours, h = cv2.findContours(
    thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

for cnt in contours:
    approx = cv2.approxPolyDP(cnt, 0.01*cv2.arcLength(cnt, True), True)
    print(len(approx))
    if len(approx) == 5:
        print("pentagon")
        cv2.drawContours(img, [cnt], 0, 255, -1)
    elif len(approx) == 3:
        print("triangle")
        cv2.drawContours(img, [cnt], 0, (0, 255, 0), -1)
    elif len(approx) == 4:
        print("square")
        cv2.drawContours(img, [cnt], 0, (0, 0, 255), -1)
    elif len(approx) == 9:
        print("half-circle")
        cv2.drawContours(img, [cnt], 0, (255, 255, 0), -1)
    elif len(approx) > 15:
        print("circle")
        cv2.drawContours(img, [cnt], 0, (0, 255, 255), -1)

cv2.imshow("test", img)
cv2.waitKey(0)
