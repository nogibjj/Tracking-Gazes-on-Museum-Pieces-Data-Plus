import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd


user_tag_coordinates = pd.read_csv("tags.csv")

# assuming a group reference image 
# or a participant reference image
# is received, this code should work agnostically


participant_reference_gaze_csv = pd.read_csv("test reference gaze updated.csv")

participant_reference_gaze_csv['ref_coordinates'] = "(" + participant_reference_gaze_csv['ref_x_pixel'].astype(str) + ',' + participant_reference_gaze_csv['ref_y_pixel'].astype(str) + ")"

def ref_coordinate_processing(gaze_reference_df):

    gaze_reference_df['ref_coordinates'] = pd.Series(zip(participant_reference_gaze_csv['ref_x_pixel'], participant_reference_gaze_csv['ref_y_pixel']))

    # add assert statements here


def gaze_in_box(gaze_reference_df_obs, tags_df):

    tag_list_from_df = user_tag_coordinates.values.tolist()

    # observation from that updated gaze csv

    name = "noise"

    smallest_center = None

    gaze_x = gaze_reference_df_obs["ref_x_pixel"]
    gaze_y = gaze_reference_df_obs["ref_y_pixel"]

    for row in tag_list_from_df:

        # 

    
