"""
Author: Aditya John (aj391)
This script contains helper functions used by the draw_heatmap.py script
"""
import sys
import os
import cv2
import pandas as pd
import numpy as np

def convert_timestamp_ns_to_ms(gaze_df, col_name="timestamp [ns]", subtract=True):
    """
    Simple function to convert the ns linux timestamp datetype to
    normal milliseconds of elapsed time
    """
    gaze_df[col_name] = pd.to_datetime(gaze_df[col_name])
    start_timestamp = gaze_df[col_name][0]
    if subtract:
        gaze_df[col_name] = gaze_df[col_name] - start_timestamp
    gaze_df[col_name] = gaze_df[col_name].astype(np.int64) / int(1e6)
    return gaze_df


def get_closest_individual_gaze_object(cap, curr_frame, gaze_df, bounding_size):
    """
    Function to look at the current timestamp and return the pixel locations
    in the gaze csv that is closest to it.
    Draws a bounding box of the specified size around that specific pixel location
    and returns the bounding box as a cropped image.
    """
    current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
    closet_value = min(
        gaze_df["timestamp [ns]"], key=lambda x: abs(x - current_timestamp)
    )

    closest_row = pd.DataFrame(
        gaze_df[gaze_df["timestamp [ns]"] == closet_value].reset_index()
    )
    x_pixel = round(closest_row["gaze x [px]"][0])
    y_pixel = round(closest_row["gaze y [px]"][0])
    # ToDo: Modify the bounding box to make it large but dynamic enough to make it smaller closer
    # to the edges of the image (if the bb size is 250 at the edges, nothings gets drawn)
    template = curr_frame[
        y_pixel - bounding_size // 2 : y_pixel + bounding_size // 2,
        x_pixel - bounding_size // 2 : x_pixel + bounding_size // 2,
    ]

    return template, closest_row


def get_closest_reference_pixel(target, template, chosen_method=1):
    """
    Compares the cropped image from the above function to the reference image
    and returns the pixel locations that most closely matches it
    """
    methods = [
        "cv2.TM_CCOEFF",
        "cv2.TM_CCOEFF_NORMED",
        "cv.TM_CCORR",
        "cv2.TM_CCORR_NORMED",
        "cv2.TM_SQDIFF",
        "cv2.TM_SQDIFF_NORMED",
    ]
    width, height, _ = template.shape[::-1]
    method = eval(methods[chosen_method])
    res = cv2.matchTemplate(target, template, method)
    _, _, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
    bottom_right = (top_left[0] + width, top_left[1] + height)
    center = (
        int((top_left[0] + bottom_right[0]) / 2),
        int((top_left[1] + bottom_right[1]) / 2),
    )
    return center


def normalize_heatmap_dict(pixel_heatmap):
    """
    Function to normalize the data between a given range to encode the
    gradient for the heatmap
    """
    EPSILON = sys.float_info.epsilon  # Smallest possible difference.
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
    minval = pixel_heatmap[min(pixel_heatmap, key=pixel_heatmap.get)]
    maxval = min(pixel_heatmap[max(pixel_heatmap, key=pixel_heatmap.get)], 255)

    for key, val in pixel_heatmap.items():
        i_f = float(val - minval) / float(maxval - minval) * (len(colors) - 1)
        ##### i_f = abs(255 - i_f)
        ##### pixel_heatmap[key] = (255, i_f, i_f)

        i, f = int(i_f // 1), i_f % 1
        if f < EPSILON:
            pixel_heatmap[key] = colors[i]
        else:
            (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i + 1]
            pixel_heatmap[key] = (
                int(r1 + f * (r2 - r1)),
                int(g1 + f * (g2 - g1)),
                int(b1 + f * (b2 - b1)),
            )
    return pixel_heatmap


def draw_heatmap_on_ref_img(pixel_heatmap, first_frame, bounding_size=3):
    """
    Function to draw the heatmap on the reference image based on the pixel locations
    """
    for key, value in pixel_heatmap.items():
        op_frame = cv2.circle(first_frame, key, bounding_size, value, 2)
    return op_frame


def create_directory(directory):
    """
    Function to create a dictionary if it doesnt exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def resample_gaze(
    gaze_df,
    timestamp_col="timestamp [ns]",
    required_cols=["gaze x [px]", "gaze y [px]"],
    resample_freq="50ms",
):
    """
    Function to look at the current timestamp and return the pixel locations
    Function that resamples the chosen timestamp column to a new frequency
    The default is to change to 50 milli seconds
    """
    gaze_df[timestamp_col] = pd.to_datetime(gaze_df[timestamp_col])
    gaze_df.set_index(timestamp_col, inplace=True)
    gaze_df = gaze_df[required_cols].resample(resample_freq).mean().reset_index()
    gaze_df[timestamp_col] = gaze_df[timestamp_col].astype(np.int64)
    return gaze_df


"""
    Using SIFT
    sift = cv2.SIFT_create()

    # Find keypoints and compute descriptors for the template and target images
    keypoints_template, descriptors_template = sift.detectAndCompute(template, None)
    keypoints_target, descriptors_target = sift.detectAndCompute(target, None)

   # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(descriptors_template,descriptors_target,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                    singlePointColor = (255,0,0),
                    matchesMask = matchesMask,
                    flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(template,keypoints_template,target,keypoints_target,matches,None,**draw_params)
    plt.imshow(img3,),plt.show()
    number = random.randint(0, 1000)
    cv2.imwrite(f"trash/{number}_matched.png", img3)
    cv2.imwrite(f"trash/{number}_template.png", template)
    cv2.imwrite(f"trash/{number}_target.png", target)


    return (0, 0)
"""
