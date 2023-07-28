import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import gc
import copy
import gc
import math


def drawfunction(event, x, y, flags, param):
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
            # coordinates_storage.extend([[name, x1, y1, x2, y2, x1, y2, x2, y1]])
            coordinates_storage.extend(
                [[name, (x1, y1), (x2, y2), (x1, y2), (x2, y1), (center_x, center_y)]]
            )

        # first coordinate is the lower right corner
        # second coordinate is the upper left corner
        elif x1 > x2 and y1 < y2:
            # coordinates_storage.extend([[name, x2, y2, x1, y1, x2, y1, x1, y2]])
            coordinates_storage.extend(
                [[name, (x2, y2), (x1, y1), (x2, y1), (x1, y2), (center_x, center_y)]]
            )

        # first coordinate is the lower left corner
        # second coordinate is the upper right corner
        elif x1 < x2 and y1 < y2:
            # coordinates_storage.extend([[name, x1, y2, x2, y1, x1, y1, x2, y2]])
            coordinates_storage.extend(
                [[name, (x1, y2), (x2, y1), (x1, y1), (x2, y2), (center_x, center_y)]]
            )

        # first coordinate is the upper right corner
        # second coordinate is the lower left corner
        elif x1 > x2 and y1 > y2:
            # coordinates_storage.extend([[name, x2, y1, x1, y2, x2, y2, x1, y1]])
            coordinates_storage.extend(
                [[name, (x2, y1), (x1, y2), (x2, y2), (x1, y1), (center_x, center_y)]]
            )


def ref_coordinate_processing(gaze_reference_df):
    gaze_reference_df["ref_coordinates"] = pd.Series(
        zip(gaze_reference_df["ref_center_x"], gaze_reference_df["ref_center_y"])
    )

    x_coordinates_from_tuple = [
        i[0] for i in gaze_reference_df["ref_coordinates"].values.tolist()
    ]

    y_coordinates_from_tuple = [
        i[1] for i in gaze_reference_df["ref_coordinates"].values.tolist()
    ]

    assert gaze_reference_df["ref_center_x"].values.tolist() == x_coordinates_from_tuple

    assert gaze_reference_df["ref_center_y"].values.tolist() == y_coordinates_from_tuple

    return gaze_reference_df


def coordinate_parser(tuple_string):
    """Parse a string tuple into a list of integers.

    The tuple must be composed of coordinates.

    Example input: '(1,2)'

    Example output: [1,2]"""

    import re

    # remove the parentheses

    parsed_tuple = re.findall(pattern="\((\d+), (\d+)\)", string=tuple_string)[0]

    formatted_tuple = [int(i) for i in parsed_tuple]

    return formatted_tuple


def gaze_tagger(gaze_reference_df_obs, tags_df):
    """Tag the observation gaze with the feature"""

    # observation from that updated gaze csv
    # obtained and parsed from ref_coordinates

    obs_x, obs_y = gaze_reference_df_obs[0], gaze_reference_df_obs[1]

    # extracting the features for the following loop operation

    features = [i for i in tags_df["name"].unique()]

    name = "noise"

    smallest_center_x, smallest_center_y = None, None

    distance_from_center = None

    for feature in features:
        # the coordinates here are the
        # points of the rectangle encapsulated
        # by the user's bounding in the
        # tagging_event.py script

        # upper left corner
        p1 = tags_df.loc[tags_df["name"] == feature, "(x1,y1)"].values.tolist()
        x1, y1 = coordinate_parser(p1[0])

        # lower right corner
        p2 = tags_df.loc[tags_df["name"] == feature, "(x2,y2)"].values.tolist()
        x2, y2 = coordinate_parser(p2[0])

        # lower left corner
        p3 = tags_df.loc[tags_df["name"] == feature, "(x3,y3)"].values.tolist()
        x3, y3 = coordinate_parser(p3[0])

        # upper right corner
        p4 = tags_df.loc[tags_df["name"] == feature, "(x4,y4)"].values.tolist()
        x4, y4 = coordinate_parser(p4[0])

        center = tags_df.loc[
            tags_df["name"] == feature, "(center_x,center_y)"
        ].values.tolist()
        center_x, center_y = coordinate_parser(center[0])

        # the if statements to check if the observation gaze
        # is within the bounds of the rectangle

        if (
            (obs_x >= x1 and obs_y <= y1)
            and (obs_x <= x2 and obs_y >= y2)
            and (obs_x >= x3 and obs_y >= y3)
            and (obs_x <= x4 and obs_y <= y4)
        ):
            if (
                name != "noise"
                and smallest_center_x != None
                and smallest_center_y != None
                and distance_from_center != None
            ):
                # if the observation gaze is within the bounds
                # of two or more rectangles, the one with the
                # smallest center is chosen

                if (
                    math.dist((obs_x, obs_y), (center_x, center_y))
                    < distance_from_center
                ):
                    name = copy.deepcopy(feature)

                    smallest_center_x, smallest_center_y = copy.deepcopy(
                        center_x
                    ), copy.deepcopy(center_y)

                    # print("new name : ", name)
                    # print("new center : ", smallest_center_x, smallest_center_y)

            else:
                name = copy.deepcopy(feature)

                smallest_center_x, smallest_center_y = copy.deepcopy(
                    center_x
                ), copy.deepcopy(center_y)

                distance_from_center = math.dist((obs_x, obs_y), (center_x, center_y))

    gc.collect()

    return name
