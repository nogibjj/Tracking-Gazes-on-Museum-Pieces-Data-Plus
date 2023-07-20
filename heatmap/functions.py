"""
Author: Aditya John (aj391), Eric Rios Soderman (ejr41)
This script contains helper functions used by the draw_heatmap.py script
"""
import sys
import os
import traceback
import cv2
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def is_single_color(
    cv2_array, save=False, name="empty", troubleshoot=False, robust=False
):
    """Verify if frame has pixel values of one color."""

    # Load the image
    # Some weird artifacting happens with blue pictures.
    # They change to orange.
    image = np.array(cv2_array, np.uint8)

    # image = Image.load(image)
    image = Image.fromarray(image, mode="RGB")

    # Convert the image to grayscale
    image = image.convert("L")

    # Get the pixel data

    pixels = list(image.getdata())

    # Check if all pixel values are the same
    is_single_color = all(pixel == pixels[0] for pixel in pixels)

    if troubleshoot:
        return pixels
    if save:
        image.save(f"{name}.png")

    if is_single_color:
        # print("The image is only of one color.")
        return True
    elif robust:
        # most_common_color = max(set(pixels), key=pixels.count)
        new_pixels = np.array(image.getdata())
        arr, counts = np.unique(pixels, return_counts=True)
        idx = np.argmax(counts)
        most_common_color = arr[idx]
        # if pixels.count(most_common_color) > len(pixels) * 0.9:
        # print("The image is mostly one color or more colors.")
        # return True
        if counts[idx] > new_pixels.shape[0] * 0.9:
            # print("The image is mostly one color or more colors.")
            return True
    else:
        # print("The image contains multiple colors.")
        return False


# abstracting important pieces to other functions

# create the sift object and its keypoints
# and descriptors by abstracting the code into
# a function


def image_matcher(reference_frame, comparison_frame):
    """Find the keypoints and descriptors with SIFT.

    Original scripts considered reading the images in
    grayscale and with cv2.imread"""

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(reference_frame, None)
    kp2, des2 = sift.detectAndCompute(comparison_frame, None)

    if kp1 == ():
        print("Reference Frame has no features to detect")
        return [], (), ()

    if kp2 == ():
        print("Comparision frame has no features to detect")
        return [], (), ()

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    return matches, kp1, kp2


# choosing the best pairs guarantees the best accuracy
# when mapping the gaze points from a random frame to
# our reference image finder's best reference image


def pair_generators(
    distance_multiplier=0.1, gaze_point=None, matches=None, kp1=None, kp2=None
):
    """Generate the list of keypoint pairs.
    These keypoints come from the SIFT algorithm output."""
    good_pairs = []
    for m, n in matches:
        if m.distance < distance_multiplier * n.distance:
            pt1 = kp1[m.queryIdx].pt
            pt1 = (int(pt1[0]), int(pt1[1]))
            pt2 = kp2[m.trainIdx].pt
            pt2 = (int(pt2[0]), int(pt2[1]))

            if pt1[0] == gaze_point[0] or pt2[0] == gaze_point[0]:
                continue

            elif pt1[1] == gaze_point[1] or pt2[1] == gaze_point[1]:
                continue

            good_pairs.append([pt1, pt2])

    return good_pairs


# loop function that finds you the ideal pair
def ideal_pair(
    dist_ranges=np.arange(0.05, 1.0, 0.05),
    gaze_point=None,
    matches=None,
    kp1=None,
    kp2=None,
):
    """Find the best pair for the gaze point.

    It will stop once it finds at least 2 pairs."""

    for value in dist_ranges:
        pairs_list = pair_generators(
            distance_multiplier=value,
            gaze_point=gaze_point,
            matches=matches,
            kp1=kp1,
            kp2=kp2,
        )
        pairs_list = list(set(tuple(sub) for sub in pairs_list))
        if len(pairs_list) >= 2:
            return pairs_list[0:2]

        else:
            pass

    return None


def keypoints_finder(
    reference_frame=None,
    comparison_frame=None,
    gaze_point=None,
    dist_ranges=np.arange(0.05, 1.0, 0.05),
):
    """Find the best two pairs of query and train points
    for the gaze point.

    This algorithm is meant to facilitate the mapping of the
    comparison gaze point to the reference image's gaze point."""

    matches, kp1, kp2 = image_matcher(
        reference_frame=reference_frame, comparison_frame=comparison_frame
    )
    if matches == []:
        return None

    pairs_list = ideal_pair(
        dist_ranges=dist_ranges,
        gaze_point=gaze_point,
        matches=matches,
        kp1=kp1,
        kp2=kp2,
    )

    if pairs_list is None:
        return None

    else:
        return pairs_list


# Geometric solution of the problem


"""
From this point onwards, two pairs of points are needed to solve the problem.
The previous functions were designed to obtain two reference - comparison pairs of points.

The reference image is the image on which we want to map the gaze point. 
The comparison image is the image from which we want to map the gaze point.
The gaze point is a point that is frame dependent.

A coordinate in one frame may not be the same coordinate 
in the reference image, because spatial changes may have
taken place across time. 

For example, one comparison frame may be more zoomed in
when compared to the reference image, or it may be more
translated to the right or left of the field of vision.

For these reasons, we needed a scale invariant algorithm
like SIFT to find common points across both images to
then estimate the gaze point's location on the reference image
by using the comparison points and gaze point
from the comparison image.

"""


def slope_finder(comparison_point, gaze_point):
    """Find the slope of the line that connects two points.

    In this case, we want the slope of the line that connects
    the gaze point and the comparison point."""

    slope = (gaze_point[1] - comparison_point[1]) / (
        gaze_point[0] - comparison_point[0]
    )
    return slope


def intercept_finder(reference_point, comparison_slope):
    """Find the intercept of the line that connects two points.

    In this case, we want the intercept of the line that connects
    the future gaze point to be mapped unto the reference image
    and the reference point. We don't have the gaze point yet,
    but we can estimate the lines intercept by using the
    comparison slope and the reference point. Those two
    lines will preserve the same relationship (slope) with
    their gaze points."""

    intercept = reference_point[1] - (comparison_slope * reference_point[0])
    return intercept


def slope_intercept_finder(reference_point, comparison_point, gaze_point):
    """Find the slope and intercept of the line that connects two points.

    The slope is obtained from the comparison point and the gaze point.

    The intercept is obtained by using that slope and the reference point.

    """

    slope = slope_finder(comparison_point, gaze_point)
    intercept = intercept_finder(reference_point, slope)

    return slope, intercept


# make a tuple from the slope and intercept


def intersecting_point(slope_intercept_a, slope_intercept_b):
    """Find the point at which two lines intersect.

    The idea for solving this problem in a "coding friendly" way
    comes from here :

    https://www.cuemath.com/geometry/intersection-of-two-lines/

    Arguments :

    - slope_intercept_a :  a tuple containing the slope and intercept of the first reference point.

    - slope_intercept_b :  a tuple containing the slope and intercept of the first reference point.

    By obtaining the slopes and intercepts, we can effectively find
    the reference gaze point."""

    # First, identify the terms by using the standard form from the
    # equations of these lines. We start with the assumption that we are
    # converting the slope-intercept form of a line to standard form.
    # This is why the coefficient for y will be -1.
    # 0 = Ax + By + C

    a_1 = slope_intercept_a[0]
    b_1 = -1  # negative and constant because of conversion
    c_1 = slope_intercept_a[1]

    a_2 = slope_intercept_b[0]
    b_2 = -1  # negative and constant because of conversion
    c_2 = slope_intercept_b[1]

    # terms
    x0_t1 = b_1 * c_2 - b_2 * c_1
    x0_t2 = a_1 * b_2 - a_2 * b_1
    x0 = x0_t1 / x0_t2

    y0_t1 = c_1 * a_2 - c_2 * a_1
    y0_t2 = a_1 * b_2 - a_2 * b_1
    y0 = y0_t1 / y0_t2

    return x0, y0


def reference_gaze_point_mapper(
    reference_frame,
    comparison_frame,
    gaze_point,
    # dist_ranges=np.arange(0.05, 1.0, 0.05),
    dist_ranges=np.arange(0.0, 1.0, 0.01),
):
    """Map the gaze point from the comparison image to the reference image.

    This function uses the slopes and intercepts of the reference
    and comparison points to map the comparison gaze point to
    a reference gaze point.
    """

    # find the paired reference and comparison points
    # the size of the list is 2
    best_pairs = keypoints_finder(
        reference_frame, comparison_frame, gaze_point, dist_ranges=dist_ranges
    )

    pair_1_ref_pt = best_pairs[0][0]
    pair_1_comparison_pt = best_pairs[0][1]

    pair_2_ref_pt = best_pairs[1][0]
    pair_2_comparison_pt = best_pairs[1][1]

    # find the slopes of the comparison lines that connect to
    # the gaze point

    slope_intercept_a = slope_intercept_finder(
        pair_1_ref_pt, pair_1_comparison_pt, gaze_point
    )

    slope_intercept_b = slope_intercept_finder(
        pair_2_ref_pt, pair_2_comparison_pt, gaze_point
    )

    # find the reference gaze point
    reference_gaze_point = intersecting_point(slope_intercept_a, slope_intercept_b)

    # assert (
    #     reference_gaze_point[1]
    #     == slope_intercept_a[0] * reference_gaze_point[0] + slope_intercept_a[1]
    # )

    # assert (
    #     reference_gaze_point[1]
    #     == slope_intercept_b[0] * reference_gaze_point[0] + slope_intercept_b[1]
    # )

    reference_gaze_point = (int(reference_gaze_point[0]), int(reference_gaze_point[1]))

    return reference_gaze_point


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
    ROOT_PATH,
    name,
    first_frame,
    DETECT_BOUNDING_SIZE,
    final_img,
    updated_gaze
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

# mse function for the reference image calculations
def mse(img1, img2):
    """Mean squared error function for
    the image calculations."""
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err / (float(h * w))
    return mse


def bucket_maker(frame_dictionary: dict, bucket_size=30):
    """Makes buckets of frames for the best frame finder function.

    The output is a list of lists with integer keys from the frame_dictionary."""
    print("Making buckets of frames for the best frame finder function...")
    frame_list = sorted(frame_dictionary.keys())

    frame_list_indexes = [marker for marker in range(0, len(frame_list))][::bucket_size]

    key_buckets = []

    for idx, mark in enumerate(frame_list_indexes):
        try:
            key_buckets.extend(
                [frame_list[frame_list_indexes[idx] : frame_list_indexes[idx + 1]]]
            )

        except:  # reached the end of the list
            key_buckets.extend([frame_list[frame_list_indexes[idx] :]])

    return key_buckets


def best_frame_finder(
    frame_dictionary: dict, frame_bucket: list[int], return_mse_list: bool = False
):
    """Finds the best frame in a group of frames by
    choosing the one with the lowest MSE.

    The output is a tuple with the best frame and its key.

    Or a tuple with the best frame, its key, and a dictionary

    of the MSEs of all the frames in the frame_bucket."""

    frame_mse = dict()
    frame_mse_with_list = dict()

    for main_key in frame_bucket:
        # print(f"The main key is {main_key}")
        copy_bucket = frame_bucket.copy()
        copy_bucket.pop(copy_bucket.index(main_key))
        mse_list = list()

        for other_key in copy_bucket:
            mse_list.extend(
                [mse(frame_dictionary[main_key], frame_dictionary[other_key])]
            )

            # Mean is the best option.
            # Mode won't work because the participant is never still.
            # Median is not necessary because it is a metric meant to
            # protect against outliers.
            # However, in a video with 1000 frames and 20 jittery/uncommon ones,
            # those 20 bad ones will be outliers in all the MSEs of the
            # good frames, but the bads will simultaneously have the worst MSE.
        frame_mse[main_key] = np.mean(mse_list.copy())
        frame_mse_with_list[main_key] = (np.mean(mse_list), mse_list.copy())
        # print(f"Obtained MSE for video frame: {main_key}")
    best_frame_num = min(frame_mse, key=frame_mse.get)
    best_frame = frame_dictionary[best_frame_num]

    if return_mse_list:
        return best_frame, best_frame_num, frame_mse_with_list

    else:
        return best_frame, best_frame_num


def reference_image_finder(
    video_path: str, return_mse_list=False, buckets=30, early_stop=False
):
    """Finds the best possible reference image in a video.

    It looks for the frame with the lowest
    mean squared error (MSE) when compared to all of its other frames.

    Buckets is 30 due to most videos being recorded at 30 fps.

    A 5 minute video will have 9000 frames.

    Then, 9000/30 = 300 buckets.

    Those 300 buckets will be broken down into

    buckets equal to the number of minutes, to finally end up

    with the last few candidate, reference frames.

    The number of minutes in a video is unimportant."""

    cap = cv2.VideoCapture(video_path)
    final_frame_dictionary_gray = dict()
    final_frame_dictionary_original = dict()
    frame_dictionary_gray = dict()
    frame_dictionary_original = dict()
    temp_frame_dictionary_original = dict()
    temp_frame_dictionary_gray = dict()
    frame_number = 0
    frame_counter = 0
    minute_frame_counter = 0
    less_than_one_minute = True
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            if is_single_color(frame, robust=True):
                print("Frame is of one color. Skipping...")
                continue
            frame_number += 1
            frame_counter += 1
            temp_frame_dictionary_original[frame_number] = frame.copy()
            gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            temp_frame_dictionary_gray[frame_number] = gray_frame.copy()

            if frame_counter == buckets:
                # print(f"Frame counter reached {buckets}, now cleaning keys...")
                # gray_keys_clean = list(temp_frame_dictionary_gray.keys())
                # for key in gray_keys_clean:
                #     if is_single_color(
                #         temp_frame_dictionary_original[key], robust=True
                #     ):
                #         temp_frame_dictionary_gray.pop(key)
                #         temp_frame_dictionary_original.pop(key)
                #         print(f"Removed key {key} from the temp frame dictionary")
                #     else:
                #         print(f"Key {key} is not single colored")
                best_bucket_frame, best_bucket_frame_num = best_frame_finder(
                    temp_frame_dictionary_gray, gray_keys_clean
                )
                print(
                    f"Best frame for bucket {frame_number-buckets}-{frame_number}: {best_bucket_frame_num}"
                )
                frame_dictionary_gray[best_bucket_frame_num] = best_bucket_frame.copy()
                frame_dictionary_original[
                    best_bucket_frame_num
                ] = temp_frame_dictionary_original[best_bucket_frame_num].copy()
                temp_frame_dictionary_gray = dict()
                temp_frame_dictionary_original = dict()
                frame_counter = 0
                minute_frame_counter += 1

            if frame_number == 150 and early_stop is True:
                return frame

            elif minute_frame_counter == 60:
                print("Minute frame counter reached 60, now cleaning keys...")
                gray_keys_clean = list(frame_dictionary_gray.keys())
                for key in gray_keys_clean:
                    if is_single_color(frame_dictionary_original[key], robust=True):
                        gray_keys_clean.pop(gray_keys_clean.index(key))
                        print(f"Removed key {key} from the gray keys list")
                    else:
                        print(f"Key {key} is not single colored")
                print("Done cleaning single colored keys...")
                print(
                    "The length of the gray keys list is now : ", len(gray_keys_clean)
                )
                best_bucket_frame, best_bucket_frame_num = best_frame_finder(
                    frame_dictionary_gray, gray_keys_clean
                )
                print(f"Best frame for bucket 60 : {best_bucket_frame_num}")
                final_frame_dictionary_gray[
                    best_bucket_frame_num
                ] = best_bucket_frame.copy()
                final_frame_dictionary_original[
                    best_bucket_frame_num
                ] = frame_dictionary_original[best_bucket_frame_num].copy()
                frame_dictionary_gray = dict()
                frame_dictionary_original = dict()
                minute_frame_counter = 0
                less_than_one_minute = False

        else:
            if (
                len(list(temp_frame_dictionary_gray.keys())) == 0
                and len(list(frame_dictionary_gray.keys())) == 0
            ):
                # if it is empty, no more frames have been registered
                print("No frames present in temp frame and frame dictionaries")

                break

            # assume a case in which temp has some frames and/or
            # frame_dictionary_gray or original has frames

            try:
                # final_list = list(temp_frame_dictionary_gray.keys())
                # for key in final_list:
                #     if is_single_color(temp_frame_dictionary_gray[key], robust=True):
                #         final_list.pop(final_list.index(key))

                final_bucket_frame, final_bucket_frame_num = best_frame_finder(
                    temp_frame_dictionary_gray, list(temp_frame_dictionary_gray.keys())
                )

                # final_bucket_frame, final_bucket_frame_num = best_frame_finder(
                #     temp_frame_dictionary_gray, final_list
                # )

                frame_dictionary_gray[
                    final_bucket_frame_num
                ] = final_bucket_frame.copy()
                frame_dictionary_original[
                    final_bucket_frame_num
                ] = temp_frame_dictionary_original[final_bucket_frame_num].copy()

                print("Excess frames present in temp frame dictionaries")

            except:
                print("Excess frames not present in temp frame dictionaries")

            if less_than_one_minute:
                print("Minute frame counter is less than 60")
                # the if statement for 60 seconds never triggered
                # so the frame dictionary may easily have 59 or less final frames.
                frame_dn_original_keys = list(frame_dictionary_original.keys())
                for key in frame_dn_original_keys:
                    if is_single_color(frame_dictionary_original[key], robust=True):
                        frame_dictionary_original.pop(key)
                        frame_dictionary_gray.pop(key)
                        print(f"Removed key {key} from the less than minute keys")
                    else:
                        print(f"Key {key} is not single colored")
                final_frame_dictionary_gray = frame_dictionary_gray.copy()
                final_frame_dictionary_original = frame_dictionary_original.copy()
                break

            try:
                # Assuming some good frames remain in frame dictionaries
                best_bucket_frame, best_bucket_frame_num = best_frame_finder(
                    frame_dictionary_gray, list(frame_dictionary_gray.keys())
                )
                print(
                    "This is the length of the keys for frame dns : ",
                    len(list(frame_dictionary_gray.keys())),
                )

                final_frame_dictionary_gray[
                    best_bucket_frame_num
                ] = best_bucket_frame.copy()
                final_frame_dictionary_original[
                    best_bucket_frame_num
                ] = frame_dictionary_original[best_bucket_frame_num].copy()
                print(
                    len(final_frame_dictionary_gray.keys()),
                    "is the amount of keys in final frame for this video",
                )
                # if video is not a minute long

                print("Done storing the video frames for MSE")
                print(
                    f"Last frame number for {video_path.split(os.sep)[-1]} : {frame_number}"
                )
            except:
                print("No good frames remain in frame dictionaries")
                print("Done storing the video frames for MSE")
                print(
                    f"Last frame number for {video_path.split(os.sep)[-1]} : {frame_number}"
                )
            break

    cv2.destroyAllWindows()
    cap.release()