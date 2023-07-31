"""
Author:  Eric Rios Soderman (ejr41)
This script contains helper functions used by the tagging tool script.
"""

import cv2


def drawfunction(event, x, y, flags, param):
    """
    Register the mouse events as 4 sets of (x,y) coordinates representing a rectangle..
    """
    global x1, y1
    drawing = None
    img = param[0]
    coordinates_storage = param[1]

    if event == cv2.EVENT_LBUTTONDBLCLK or event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
    elif event == cv2.EVENT_RBUTTONDBLCLK or event == cv2.EVENT_RBUTTONDOWN:
        drawing = False
        cv2.rectangle(img, (x1, y1), (x, y), (0, 255, 0), 3)
        name = input("What is the name of the feature you want to tag?   ")
        cv2.putText(
            img=img,
            text=name,
            org=(x1, y1),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=2,
            color=(0, 255, 0),
            thickness=3,
        )
        x2 = x
        y2 = y
        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        # left upper corner is (x1, y1)
        # right lower corner is (x, y)
        if x1 < x2 and y1 > y2:
            coordinates_storage.extend(
                [[name, (x1, y1), (x2, y2), (x1, y2), (x2, y1), (center_x, center_y)]]
            )

        # first coordinate is the lower right corner
        # second coordinate is the upper left corner
        elif x1 > x2 and y1 < y2:
            coordinates_storage.extend(
                [[name, (x2, y2), (x1, y1), (x2, y1), (x1, y2), (center_x, center_y)]]
            )

        # first coordinate is the lower left corner
        # second coordinate is the upper right corner
        elif x1 < x2 and y1 < y2:
            coordinates_storage.extend(
                [[name, (x1, y2), (x2, y1), (x1, y1), (x2, y2), (center_x, center_y)]]
            )

        # first coordinate is the upper right corner
        # second coordinate is the lower left corner
        elif x1 > x2 and y1 > y2:
            coordinates_storage.extend(
                [[name, (x2, y1), (x1, y2), (x2, y2), (x1, y1), (center_x, center_y)]]
            )
