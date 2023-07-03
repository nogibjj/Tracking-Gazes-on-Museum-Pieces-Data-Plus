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

        return template, closest_row
    except:
        print(traceback.print_exc())
        return np.array([]), pd.Series()


def get_closest_reference_pixel(target, template, chosen_method=1):
    """
    Compares the cropped image from the above function to the reference image
    and returns the pixel locations that most closely matches it
    """
    try:
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

    except:
        print("template shape: ", template.shape, "target shape: ", target.shape)
        print(traceback.print_exc())
        return None


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
                (r1, g1, b1), (r2, g2, b2) = colors[i], colors[i + 1]
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


def resample_gaze(
    gaze_df,
    timestamp_col="heatmap_ts",
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


def save_outputs(
    ROOT_PATH,
    name,
    first_frame,
    DETECT_BOUNDING_SIZE,
    final_img,
    updated_gaze,
    TEMP_OUTPUT_DIR,
):
    """
    Function to save all the required outputs in a temp and original folder
    """
    ### Write the outputs to the original data folder
    cv2.imwrite(
        os.path.join(ROOT_PATH, f"{name}/reference_image_{name}.png"), first_frame
    )
    cv2.imwrite(
        os.path.join(
            ROOT_PATH,
            f"{name}/heatmap_output_{name}_{DETECT_BOUNDING_SIZE}.png",
        ),
        final_img,
    )
    updated_gaze.to_csv(
        os.path.join(ROOT_PATH, f"{name}/updated_gaze_{name}.csv"), index=False
    )

    ### Write the data to the temp output folder
    cv2.imwrite(f"{TEMP_OUTPUT_DIR}/{name}_reference_image.png", first_frame)
    cv2.imwrite(f"{TEMP_OUTPUT_DIR}/{name}_heatmap.png", final_img)
    updated_gaze.to_csv(f"{TEMP_OUTPUT_DIR}/{name}_updated_gaze.csv", index=False)


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
        print(f"Obtained MSE for video frame: {main_key}")
    best_frame_num = min(frame_mse, key=frame_mse.get)
    best_frame = frame_dictionary[best_frame_num]

    if return_mse_list:
        return best_frame, best_frame_num, frame_mse_with_list

    else:
        return best_frame, best_frame_num


def reference_image_finder(video_path: str, return_mse_list=False, buckets=30):
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
    frame_dictionary_gray = dict()
    frame_dictionary_original = dict()
    frame_number = 0
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_number += 1
            frame_dictionary_original[frame_number] = frame.copy()
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_dictionary_gray[frame_number] = gray_frame.copy()

        else:
            print("Done storing the video frames for MSE")
            print(
                f"Last frame number for {video_path.split(os.sep)[-1]} : {frame_number}"
            )
            break

    cv2.destroyAllWindows()
    cap.release()

    # obtaining the buckets for computational efficiency

    # The purpose of using the 30 as an argument in bucket_maker
    # is to eventually obtain the best frame per second.
    # This argument can be changed and should be the frame rate of the video.

    thirty_fps_buckets = bucket_maker(frame_dictionary_gray, bucket_size=buckets)

    # finding the best frame in each bucket

    best_frames_per_second = dict()

    for bucket in thirty_fps_buckets:
        best_bucket_frame, best_bucket_frame_num = best_frame_finder(
            frame_dictionary_gray, bucket
        )
        best_frames_per_second[best_bucket_frame_num] = best_bucket_frame.copy()

    # finding the best frame per minute
    # Change the bucket size to 60,
    # because there are 60 seconds in a minute.
    # And the best frame per second is already found.
    # So, one best frame will be chosen once more to represent
    # the best frame per minute.
    # This is done to reduce the number of frames to be processed
    # In addition, don't change the bucket size in the following
    # function call, since it represents the number of seconds
    # in a minute.

    sixty_seconds_buckets = bucket_maker(best_frames_per_second, bucket_size=60)

    # finding the best frame per minute

    best_frames_per_minute = dict()

    for bucket in sixty_seconds_buckets:
        best_bucket_frame, best_bucket_frame_num = best_frame_finder(
            best_frames_per_second, bucket
        )
        best_frames_per_minute[best_bucket_frame_num] = best_bucket_frame.copy()

    # finding the final best frame to become the reference frame

    _, reference_frame_num = best_frame_finder(
        best_frames_per_minute, list(best_frames_per_minute.keys())
    )

    reference_frame_original = frame_dictionary_original[reference_frame_num]

    return reference_frame_original


# def reference_image_finder(video_path: str, return_mse_list=False, buckets=30):
#     """Finds the best possible reference image in a video.

#     It looks for the frame with the lowest
#     mean squared error (MSE) when compared to all of its other frames.

#     Buckets is 30 due to most videos being recorded at 30 fps.
#     It can be modified to any number.

#     A 5 minute video will have 9000 frames.

#     Then, 9000/30 = 300 buckets.

#     Those 300 buckets will be broken down into

#     buckets equal to the number of minutes, to finally end up

#     with the last few candidate, reference frames.

#     The number of minutes in a video is unimportant."""

#     cap = cv2.VideoCapture(video_path)
#     frame_dictionary_gray = dict()
#     frame_dictionary_original = dict()
#     frame_number = 0
#     while cap.isOpened():
#         success, frame = cap.read()

#         if success:
#             frame_number += 1
#             frame_dictionary_original[frame_number] = frame.copy()
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             frame_dictionary_gray[frame_number] = gray_frame.copy()

#         else:
#             print("Done storing the video frames for MSE")
#             print(
#                 f"Last frame number for {video_path.split(os.sep)[-1]} : {frame_number}"
#             )
#             break

#     cv2.destroyAllWindows()
#     cap.release()

#     ordered_keys = sorted(frame_dictionary_gray.keys())
#     ordered_keys = ordered_keys[::buckets]

#     for key in ordered_keys:
#         frame_mse = dict()
#         frame_mse_with_list = dict()

#         for main_key in frame_dictionary_gray:
#             copy_dn = frame_dictionary_gray.copy()
#             copy_dn.pop(main_key)
#             mse_list = list()

#             for other_key in copy_dn:
#                 mse_list.extend(
#                     [mse(frame_dictionary_gray[main_key], copy_dn[other_key])]
#                 )

#             # Mean is the best option.
#             # Mode won't work because the participant is never still.
#             # Median is not necessary because it is a metric meant to
#             # protect against outliers.
#             # However, in a video with 1000 frames and 20 jittery/uncommon ones,
#             # those 20 bad ones will be outliers in all the MSEs of the
#             # good frames, but the bads will simultaneously have the worst MSE.
#             frame_mse[main_key] = np.mean(mse_list.copy())
#             frame_mse_with_list[main_key] = (np.mean(mse_list), mse_list.copy())
#             print(f"Obtained MSE for video frame: {main_key}")
#         best_frame_num = min(frame_mse, key=frame_mse.get)
#         best_frame = frame_dictionary_original[best_frame_num]

#     frame_mse = dict()
#     frame_mse_with_list = dict()

#     for main_key in frame_dictionary_gray:
#         copy_dn = frame_dictionary_gray.copy()
#         copy_dn.pop(main_key)
#         mse_list = list()

#         for other_key in copy_dn:
#             mse_list.extend([mse(frame_dictionary_gray[main_key], copy_dn[other_key])])

#         # Mean is the best option.
#         # Mode won't work because the participant is never still.
#         # Median is not necessary because it is a metric meant to
#         # protect against outliers.
#         # However, in a video with 1000 frames and 20 jittery/uncommon ones,
#         # those 20 bad ones will be outliers in all the MSEs of the
#         # good frames, but the bads will simultaneously have the worst MSE.
#         frame_mse[main_key] = np.mean(mse_list.copy())
#         frame_mse_with_list[main_key] = (np.mean(mse_list), mse_list.copy())
#         print(f"Obtained MSE for video frame: {main_key}")
#     best_frame_num = min(frame_mse, key=frame_mse.get)
#     best_frame = frame_dictionary_original[best_frame_num]

#     print(f"Obtained best frame for video : {best_frame_num}")

#     if return_mse_list:
#         return best_frame, frame_mse_with_list

#     else:
#         return best_frame


# def reference_image_finder(video_path: str, return_mse_list=False, buckets=30):
#     """Finds the best possible reference image in a video.

#     It looks for the frame with the lowest
#     mean squared error (MSE) when compared to all of its other frames.

#     Buckets is 30 due to most videos being recorded at 30 fps."""

#     # mse function for the image calculations
#     def mse(img1, img2):
#         h, w = img1.shape
#         diff = cv2.subtract(img1, img2)
#         err = np.sum(diff**2)
#         mse = err / (float(h * w))
#         return mse

#     cap = cv2.VideoCapture(video_path)
#     frame_dictionary_gray = dict()
#     frame_dictionary_original = dict()
#     frame_number = 0
#     while cap.isOpened():
#         success, frame = cap.read()

#         if success:
#             frame_number += 1
#             frame_dictionary_original[frame_number] = frame.copy()
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             frame_dictionary_gray[frame_number] = gray_frame.copy()

#         else:
#             print("Done storing the video frames for MSE")
#             print(
#                 f"Last frame number for {video_path.split(os.sep)[-1]} : {frame_number}"
#             )
#             break

#     cv2.destroyAllWindows()
#     cap.release()

#     ordered_keys = sorted(frame_dictionary_gray.keys())
#     ordered_keys = ordered_keys[::buckets]

#     for key in ordered_keys:
#         frame_mse = dict()
#         frame_mse_with_list = dict()

#         for main_key in frame_dictionary_gray:
#             copy_dn = frame_dictionary_gray.copy()
#             copy_dn.pop(main_key)
#             mse_list = list()

#             for other_key in copy_dn:
#                 mse_list.extend(
#                     [mse(frame_dictionary_gray[main_key], copy_dn[other_key])]
#                 )

#             # Mean is the best option.
#             # Mode won't work because the participant is never still.
#             # Median is not necessary because it is a metric meant to
#             # protect against outliers.
#             # However, in a video with 1000 frames and 20 jittery/uncommon ones,
#             # those 20 bad ones will be outliers in all the MSEs of the
#             # good frames, but the bads will simultaneously have the worst MSE.
#             frame_mse[main_key] = np.mean(mse_list.copy())
#             frame_mse_with_list[main_key] = (np.mean(mse_list), mse_list.copy())
#             print(f"Obtained MSE for video frame: {main_key}")
#         best_frame_num = min(frame_mse, key=frame_mse.get)
#         best_frame = frame_dictionary_original[best_frame_num]

#     frame_mse = dict()
#     frame_mse_with_list = dict()

#     for main_key in frame_dictionary_gray:
#         copy_dn = frame_dictionary_gray.copy()
#         copy_dn.pop(main_key)
#         mse_list = list()

#         for other_key in copy_dn:
#             mse_list.extend([mse(frame_dictionary_gray[main_key], copy_dn[other_key])])

#         # Mean is the best option.
#         # Mode won't work because the participant is never still.
#         # Median is not necessary because it is a metric meant to
#         # protect against outliers.
#         # However, in a video with 1000 frames and 20 jittery/uncommon ones,
#         # those 20 bad ones will be outliers in all the MSEs of the
#         # good frames, but the bads will simultaneously have the worst MSE.
#         frame_mse[main_key] = np.mean(mse_list.copy())
#         frame_mse_with_list[main_key] = (np.mean(mse_list), mse_list.copy())
#         print(f"Obtained MSE for video frame: {main_key}")
#     best_frame_num = min(frame_mse, key=frame_mse.get)
#     best_frame = frame_dictionary_original[best_frame_num]

#     print(f"Obtained best frame for video : {best_frame_num}")

#     if return_mse_list:
#         return best_frame, frame_mse_with_list

#     else:
#         return best_frame


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
