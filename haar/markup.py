"""
Module with definitions for marking up images such as:
- Drawing superimposed shapes
- Adding superimposed text
"""

import cv2


def label_court(img_rgb, x_pos, y_pos, width, height):
    """
    Draw rectangle at given positions and dimensions, and label it a tennis court
    """
    cv2.rectangle(img_rgb, (x_pos, y_pos),
                  (x_pos + width, y_pos + height),
                  (0, 255, 0), 5)
    cv2.putText(img_rgb, "tennis", (x_pos + int((width/8)), y_pos + int((height/3))),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_rgb, "court", (x_pos + int((width/8)), y_pos + 30 + int((height/3))),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
