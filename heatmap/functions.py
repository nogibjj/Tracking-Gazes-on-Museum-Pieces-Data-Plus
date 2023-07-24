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
    input_cv2_array,
    resize=(100, 100),
    maximum_robust=False,
):
    """Verify if frame has pixel values of one color.

    maximum_robust is a flag that will not allow the function
    to resize the image. The tradeoff is maximum accuracy for
    slowest speed."""
    if not (maximum_robust):
        # Other method is INTER_AREA
        cv2_array = cv2.resize(input_cv2_array, resize, interpolation=cv2.INTER_CUBIC)

    image = np.array(cv2_array, np.uint8)

    image = Image.fromarray(image, mode="RGB")

    # Convert the image to grayscale
    image = image.convert("L")

    if maximum_robust:
        # this is the first pass of the function
        # to save computation time
        # Get the pixel data

        pixels = list(image.getdata())

        # Check if all pixel values are the same
        single_color = all(pixel == pixels[0] for pixel in pixels)

        if single_color:
            # print("The image is only of one color.")
            return True

    # this is the fully robust computation below
    # but it is very slow with large arrays
    new_pixels = np.array(image.getdata())
    _, counts = np.unique(new_pixels, return_counts=True)
    idx = np.argmax(counts)

    if counts[idx] > new_pixels.shape[0] * 0.9:
        # print("The image is mostly one color or more colors.")
        return True
    else:
        # print("The image contains multiple colors.")
        return False


# Abstracting image shifting algorithm's pieces to other functions

"""
https://blog.francium.tech/feature-detection-and-matching-with-opencv-5fd2394a590

"""


def reference_gaze_point_mapper(img1, img2, x_pixel, y_pixel):
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
    video_path: str, buckets=30, early_stop=False, resize_factor=(500, 500), debug=False
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
            if debug:
                is_single_color(frame, maximum_robust=True)

            else:
                if is_single_color(frame, resize=resize_factor):
                    print("Frame is of one color. Skipping...")
                    continue
            frame_number += 1
            frame_counter += 1
            temp_frame_dictionary_original[frame_number] = frame.copy()
            gray_frame = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
            temp_frame_dictionary_gray[frame_number] = gray_frame.copy()

            if frame_counter == buckets:
                best_bucket_frame, best_bucket_frame_num = best_frame_finder(
                    temp_frame_dictionary_gray, list(temp_frame_dictionary_gray.keys())
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
                best_bucket_frame, best_bucket_frame_num = best_frame_finder(
                    frame_dictionary_gray, list(frame_dictionary_gray.keys())
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
                final_bucket_frame, final_bucket_frame_num = best_frame_finder(
                    temp_frame_dictionary_gray, list(temp_frame_dictionary_gray.keys())
                )

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
                final_frame_dictionary_gray = frame_dictionary_gray.copy()
                final_frame_dictionary_original = frame_dictionary_original.copy()
                del frame_dictionary_gray
                del frame_dictionary_original
                break

            try:
                # Assuming some good frames remain in frame dictionaries
                best_bucket_frame, best_bucket_frame_num = best_frame_finder(
                    frame_dictionary_gray, list(frame_dictionary_gray.keys())
                )

                final_frame_dictionary_gray[
                    best_bucket_frame_num
                ] = best_bucket_frame.copy()
                final_frame_dictionary_original[
                    best_bucket_frame_num
                ] = frame_dictionary_original[best_bucket_frame_num].copy()

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

    # finding the final best frame to become the reference frame

    reference_frame_gray, reference_frame_num = best_frame_finder(
        final_frame_dictionary_gray, list(final_frame_dictionary_gray.keys())
    )

    reference_frame_original = final_frame_dictionary_original[reference_frame_num]

    print(f"Obtained best frame for video : {reference_frame_num}")

    return reference_frame_original, reference_frame_gray
