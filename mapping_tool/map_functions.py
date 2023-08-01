"""
Author: Eric Rios Soderman (ejr41)
This script contains helper functions used by the mapping script.
"""
import pandas as pd
import gc
import copy
import gc
import math


def ref_coordinate_processing(gaze_reference_df):
    """Process the reference coordinates into a tuple"""

    # remove rows with nan values in the ref_center_x and ref_center_y columns

    gaze_reference_df = (
        gaze_reference_df.dropna(subset=["ref_center_x", "ref_center_y"])
        .reset_index(drop=True)
        .copy()
    )

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
    """Tag the observation gaze point with the most likely feature.

    The feature with the closest center point to the observation gaze point
    will be chosen as the tag for that gaze point."""

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

            else:
                name = copy.deepcopy(feature)

                smallest_center_x, smallest_center_y = copy.deepcopy(
                    center_x
                ), copy.deepcopy(center_y)

                distance_from_center = math.dist((obs_x, obs_y), (center_x, center_y))

    # for cleanup
    gc.collect()

    return name