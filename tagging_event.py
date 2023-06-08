import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


drawing = True

feature_coordinates = []


def drawfunction(event, x, y, flags, param):
    global x1, y1
    if event == cv2.EVENT_LBUTTONDBLCLK or event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x1, y1 = x, y
    elif event == cv2.EVENT_RBUTTONDBLCLK or event == cv2.EVENT_RBUTTONDOWN:
        drawing = False
        cv2.rectangle(img, (x1, y1), (x, y), (0, 255, 0), 3)
        name = input("What is the name of the feature you are interested in?   ")
        cv2.putText(
            img=img,
            text=name,
            org=(x1, y1),
            fontFace=cv2.FONT_HERSHEY_TRIPLEX,
            fontScale=4,
            color=(0, 255, 0),
            thickness=3,
        )
        x2 = x
        y2 = y
        # left upper corner is (x1, y1)
        # right lower corner is (x, y)
        if x1 < x2 and y1 > y2:
            feature_coordinates.extend([[name, x1, y1, x2, y2, x1, y2, x2, y1]])

        # first coordinate is the lower right corner
        # second coordinate is the upper left corner
        elif x1 > x2 and y1 < y2:
            feature_coordinates.extend([[name, x2, y2, x1, y1, x2, y1, x1, y2]])

        # first coordinate is the lower left corner
        # second coordinate is the upper right corner
        elif x1 < x2 and y1 < y2:
            feature_coordinates.extend([[name, x1, y2, x2, y1, x1, y1, x2, y2]])

        # first coordinate is the upper right corner
        # second coordinate is the lower left corner
        elif x1 > x2 and y1 > y2:
            feature_coordinates.extend([[name, x2, y1, x1, y2, x2, y2, x1, y1]])


# base_img = cv2.imread("test2 image prompter.png")
# base_img = cv2.imread("test3 image prompter.jpg")
base_img = cv2.imread("test5 image prompter.jpg")
# base_img = cv2.imread("test4 image prompter.jpg")
# img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
img = base_img
reset_img = img.copy()

plt.imshow(img)

cv2.namedWindow("image")

cv2.setMouseCallback("image", drawfunction)
flag = True
while flag:
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("0"):
        break

    elif key == ord("5"):
        img = reset_img.copy()
        cv2.imshow("image", img)
        print("You have reset the image")

    elif key == ord("9"):
        flag = False
        print("You have finished tagging")
# cv2.imshow("image", img)

cv2.destroyAllWindows()

# For the final dataframe
# x1 and y1 correspond to the upper left corner
# x2 and y2 correspond to the lower right corner
# x3 and y3 correspond to the lower left corner
# x4 and y4 correspond to the upper right corner
coordinates_df = pd.DataFrame(
    feature_coordinates,
    columns=["name", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"],
)

print(coordinates_df)

coordinates_df.to_csv("tags.csv", index=False)
