"""
Author: Aditya John (aj391)
This script contains helper functions used by the draw_heatmap.py script
"""
import os
import cv2
import pandas as pd
import numpy as np

def convert_timestamp_ns_to_ms(gaze_df, col_name='timestamp [ns]', subtract=False):
    """
    Simple function to convert the ns linux timestamp datetype to normal milliseconds of elapsed time
    """
    gaze_df[col_name] = pd.to_datetime(gaze_df[col_name])
    start_timestamp = gaze_df[col_name][0]
    if subtract:
        gaze_df[col_name] = (gaze_df[col_name] - start_timestamp)
    gaze_df[col_name] = gaze_df[col_name].astype(np.int64) / int(1e6)
    return gaze_df

def get_closest_individual_gaze_object(cap, curr_frame, gaze_csv, bounding_size):
    """
    Function to look at the current timestamp and return the pixel locations in the gaze csv that is closest to it. 
    Draws a bounding box of the specified size around that specific pixel location and returns the bounding box as a cropped image. 
    """
    current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    print(current_timestamp)
    closet_value = min(gaze_csv['timestamp [ns]'], key=lambda x:abs(x-current_timestamp))
    closest_row = pd.DataFrame(gaze_csv[gaze_csv['timestamp [ns]'] == closet_value].reset_index())
    x_pixel = round(closest_row['gaze x [px]'][0])
    y_pixel = round(closest_row['gaze x [px]'][0])
    template = curr_frame[y_pixel:y_pixel+bounding_size, x_pixel:x_pixel+bounding_size]
    return template, closest_row

def get_closest_reference_pixel(first_frame, template, method=1):
    """
    Compares the cropped image from the above function to the reference image (likely the first non-grey frame) 
    and returns the pixel locations that most closely matches it
    """
    chosen_method = 1
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    method = eval(methods[chosen_method])
    res = cv2.matchTemplate(first_frame, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    return top_left

def normalize_heatmap_dict(pixel_heatmap, new_min=0, new_max=1):
    """
    Function to normalize the heatmap dictionary counts between the given range
    """
    old_min = pixel_heatmap[min(pixel_heatmap, key=pixel_heatmap.get)]
    old_max = pixel_heatmap[max(pixel_heatmap, key=pixel_heatmap.get)]
    new_min = new_min
    new_max = new_max
    old_range = (old_max - old_min)  
    new_range = (new_max - new_min) 
    
    for key, value in pixel_heatmap.items():
        color = (((value - old_min) * new_range) / old_range) + new_min
        pixel_heatmap[key] = color

def draw_heatmap_on_ref_img(pixel_heatmap, first_frame, bounding_size):

    """
    Function to draw the heatmap on the reference image based on the pixel locations
    """
    """ #### Not working
    overlay = first_frame.copy()
    ### Translucent small circle bounding box
    for key, value in pixel_heatmap.items():
        bottom_right = (key[0] + bounding_size, key[1] + bounding_size)
        #value = abs(255-value)
        color = (255, 255, 255)
        op_frame = cv2.circle(first_frame, key, 3, color, -1)
        alpha = 1-value  # Transparency factor.
        op_frame = cv2.addWeighted(overlay, alpha, op_frame, 1 - alpha, 0)
    return op_frame
    """


    ### Small circle bounding box
    for key, value in pixel_heatmap.items():
        bottom_right = (key[0] + bounding_size, key[1] + bounding_size)
        value = abs(255-value)
        color = (value, value, 255)
        op_frame = cv2.circle(first_frame, key, 3, color, 2)
    return op_frame
    

    ### Rectangle bounding box version
    """
    for key, value in pixel_heatmap.items():
        bottom_right = (key[0] + bounding_size, key[1] + bounding_size)
        value = abs(255-value)
        color = (value, value, 255)
        op_frame = cv2.rectangle(first_frame, key, bottom_right, color, 2) 
    return op_frame
    """
def create_output_directory():
    if not os.path.exists('./output'):
        os.makedirs('./output')