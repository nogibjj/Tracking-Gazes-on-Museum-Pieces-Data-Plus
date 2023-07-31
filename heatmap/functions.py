"""
Author: Aditya John (aj391), Eric Rios Soderman (ejr41)
This script contains helper functions used by the draw_heatmap.py script.
"""
import sys
import os
import traceback
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# Abstracting image shifting algorithm's pieces to other functions

"""
https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590

"""


def reference_gaze_point_mapper(img1, img2, x_pixel, y_pixel):
    """
    Map the gaze point from the current image to the reference image.

    The main idea is to find the homography matrix that transforms the
    gaze point from the current image to the reference image.

    """
    gaze_point = np.array([[x_pixel, y_pixel, 1]]).reshape(3, 1)
    MIN_MATCHES = 5

    orb = cv2.ORB_create(nfeatures=2500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=2)
    search_params = {}
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # As per Lowe's ratio test to filter good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCHES:
        src_points = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        dst_points = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(
            -1, 1, 2
        )
        m, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        transformed_pixel = np.dot(m, gaze_point)
        transformed_x = round((transformed_pixel[0] / transformed_pixel[2])[0])
        transformed_y = round((transformed_pixel[1] / transformed_pixel[2])[0])
        return (transformed_x, transformed_y)
    else:
        print("No matches found")
    return (None, None)


def get_closest_individual_gaze_object(
    cap, curr_frame, gaze_df, bounding_size, timestamp_col="heatmap_ts"
):
    """
    Function to look at the current timestamp and return the pixel locations
    in the gaze csv that is closest to it.
    Draws a bounding box of the specified size around that specific pixel location
    and returns the bounding box as a cropped image.
    """
    try:
        current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        closet_value = min(
            gaze_df[timestamp_col], key=lambda x: abs(x - current_timestamp)
        )

        closest_row = pd.DataFrame(
            gaze_df[gaze_df[timestamp_col] == closet_value].reset_index()
        )
        x_pixel = round(closest_row["gaze x [px]"][0])
        y_pixel = round(closest_row["gaze y [px]"][0])
        # ToDo: Modify the bounding box to make it large but dynamic enough to make it smaller closer
        # to the edges of the image (if the bb size is 250 at the edges, nothings gets drawn)
        template = curr_frame[
            y_pixel - bounding_size // 2 : y_pixel + bounding_size // 2,
            x_pixel - bounding_size // 2 : x_pixel + bounding_size // 2,
        ]

        return template, closest_row, x_pixel, y_pixel
    except:
        print(traceback.print_exc())
        return np.array([]), pd.Series()


def normalize_heatmap_dict(pixel_heatmap):
    """
    Function to normalize the data between a given range to encode the
    gradient for the heatmap
    """
    try:
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
                (r1, g1, b1), (r2, g2, b2) = colors[i % 3], colors[(i + 1) % 3]
                pixel_heatmap[key] = (
                    int(r1 + f * (r2 - r1)),
                    int(g1 + f * (g2 - g1)),
                    int(b1 + f * (b2 - b1)),
                )
        return pixel_heatmap
    except:
        print(traceback.print_exc())
        return pixel_heatmap


def draw_heatmap_on_ref_img(pixel_heatmap, first_frame, bounding_size=3):
    """
    Function to draw the heatmap on the reference image based on the pixel locations
    """
    try:
        for key, value in pixel_heatmap.items():
            op_frame = cv2.circle(first_frame, key, bounding_size, value, 2)
        return op_frame
    except:
        print(traceback.print_exc())
        return first_frame


def create_directory(directory):
    """
    Function to create a dictionary if it doesnt exist
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_outputs(
    ROOT_PATH, name, first_frame, DETECT_BOUNDING_SIZE, final_img, updated_gaze
):
    """
    Function to save all the required outputs in a temp and original folder
    """
    ### Write the outputs to the original data folder
    cv2.imwrite(
        os.path.join(ROOT_PATH, f"reference_image_{name}_SIFT.png"), first_frame
    )
    cv2.imwrite(
        os.path.join(
            ROOT_PATH,
            f"heatmap_output_{name}_{DETECT_BOUNDING_SIZE}_SIFT.png",
        ),
        final_img,
    )
    updated_gaze.to_csv(
        os.path.join(ROOT_PATH, f"updated_gaze_{name}_SIFT.csv"), index=False
    )
