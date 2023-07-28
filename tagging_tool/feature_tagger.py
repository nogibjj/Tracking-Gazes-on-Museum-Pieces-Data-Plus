"""This is a script to tag the features in the image. 

The script will save the coordinates of the features in a csv file.

Author: Eric Rios-Soderman"""

import os
from tag_event_functions import drawfunction
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import sys
import copy

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from config.config import *

# print(" Made it past imports")
# Set env variables based on config file
try:
    env = sys.argv[1]
    env_var = eval(env + "_config")
except:
    print("Enter valid env variable. Refer to classes in the config.py file")
    sys.exit()


# the path will be the same, regardless if supplied or generated
data_folder_path = os.path.join(
    env_var.ROOT_PATH, env_var.ART_PIECE
)  # all the participant folders are here

reference_image = cv2.imread(
    os.path.join(env_var.ROOT_PATH, env_var.ART_PIECE, "reference_image.png")
)

output_folder_path = os.path.join(env_var.OUTPUT_PATH, env_var.ART_PIECE)

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)


feature_coordinates = []
drawing = True
flag = True

param = [reference_image, feature_coordinates]

img = reference_image
reset_img = copy.deepcopy(reference_image)

# get the resolution of the image
height, width, channels = img.shape
print(f"width: {width}, height: {height}, channels: {channels}")

cv2.namedWindow("image", flags=cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", width, height)

cv2.setMouseCallback("image", drawfunction, param)

while flag:
    cv2.imshow("image", img)
    key = cv2.waitKey(1)
    if key == ord("0"):
        break

    elif key == ord("5"):
        cv2.destroyAllWindows()
        img = copy.deepcopy(reset_img)
        print("You have reset the image")
        cv2.namedWindow("image", flags=cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", width, height)
        feature_coordinates = []
        param = [img, feature_coordinates]
        cv2.setMouseCallback("image", drawfunction, param)

    elif key == ord("9"):
        flag = False
        print("You have finished tagging")

cv2.destroyAllWindows()

coordinates_df = pd.DataFrame(
    feature_coordinates,
    columns=[
        "name",
        "(x1,y1)",
        "(x2,y2)",
        "(x3,y3)",
        "(x4,y4)",
        "(center_x,center_y)",
    ],
)

assert [i[0] for i in coordinates_df["(x1,y1)"]] == [
    i[0] for i in coordinates_df["(x3,y3)"]
]
assert [i[0] for i in coordinates_df["(x2,y2)"]] == [
    i[0] for i in coordinates_df["(x4,y4)"]
]
assert [i[1] for i in coordinates_df["(x1,y1)"]] == [
    i[1] for i in coordinates_df["(x4,y4)"]
]
assert [i[1] for i in coordinates_df["(x2,y2)"]] == [
    i[1] for i in coordinates_df["(x3,y3)"]
]

print("assertions passed")
print(coordinates_df.head())

# current_time = dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

print(f"Saving the tag coordinates csv file")
# coordinates_df.to_csv(
#     os.path.join(output_folder_path, f"tags_coordinates_{current_time}.csv"),
#     index=False,
# )
coordinates_df.to_csv(
    os.path.join(output_folder_path, "tags_coordinates.csv"),
    index=False,
)

print(f"Finished generating tags")
