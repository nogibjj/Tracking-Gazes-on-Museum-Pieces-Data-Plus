"""
Author: Eric Rios Soderman (ejr41)
This script contains helper functions used by the draw_heatmap.py script.
"""
import os
import cv2
import numpy as np
from PIL import Image
import array as arr


def is_single_color(
    input_cv2_array,
    resize=(100, 100),
    maximum_robust=False,
):
    """Verify if frame has pixel values of mostly one color.

    maximum_robust is a flag that will not allow the function
    to resize the image. The tradeoff is maximum accuracy for
    slowest speed."""
    if maximum_robust:
        cv2_array = input_cv2_array

    else:
        # Other method is INTER_AREA; no discernable difference was noticed,
        # but the checking of this was not rigorous.
        cv2_array = cv2.resize(input_cv2_array, resize, interpolation=cv2.INTER_CUBIC)

    image = np.array(cv2_array, np.uint8)

    image = Image.fromarray(image, mode="RGB")

    # Convert the image to grayscale
    image = image.convert("L")

    if maximum_robust:
        # this is the first pass of the function
        # to save computation time, as most
        # single color frames will be caught here.

        # Get the pixel data

        pixels = list(image.getdata())

        # Check if all pixel values are the same
        single_color = all(pixel == pixels[0] for pixel in pixels)

        if single_color:
            # print("The image is only of one color.")
            return True

    # This is the fully robust computation below,
    # but it is very slow with large arrays.
    # The benefit is that it will catch frames
    # that are not single color, but are close to it,
    # such as a frame with a lot of white and a little black.
    new_pixels = np.array(image.getdata())
    _, counts = np.unique(new_pixels, return_counts=True)
    idx = np.argmax(counts)

    # 90% of the pixels are of the same color.
    # This is a very high threshold, but it is necessary
    # to avoid false positives.
    if counts[idx] > new_pixels.shape[0] * 0.9:
        # print("The image is mostly one color or more colors.")
        return True
    else:
        # print("The image contains multiple colors.")
        return False


# mse function for the reference image calculations
def mse(img1, img2, debug=False):
    """Mean squared error function for
    the image calculations.

    This function is mainly used on grayscale images."""
    if debug:
        print("img1 shape: ", img1.shape)
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
            # good frames, but the bad ones will simultaneously have the worst MSE.
        frame_mse[main_key] = np.mean(mse_list.copy())
        frame_mse_with_list[main_key] = (np.mean(mse_list), mse_list.copy())
        # print(f"Obtained MSE for video frame: {main_key}")
    best_frame_num = min(frame_mse, key=frame_mse.get)
    best_frame = frame_dictionary[best_frame_num]
    # print(f"frame COMPENDIUM: {frame_mse_with_list}")

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

    The number of minutes in a video is unimportant.

    The output is a tuple with the best frame and its key.


    Sidenote : Code is commented because experiments to check the validity of the
    reference frame after downscaling the image were conducted. They will be left there
    for future reference."""

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
            frame_number += 1
            if debug:
                if is_single_color(frame, maximum_robust=True):
                    print("Frame is of one color. Skipping...")
                    continue

            else:
                if is_single_color(frame, resize=resize_factor):
                    print("Frame is of one color. Skipping...")
                    continue
                # else:
                #     frame = cv2.resize(
                #         frame, resize_factor, interpolation=cv2.INTER_CUBIC
                #     )  # INTER_AREA is another option

            # frame_number += 1
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
                # if debug:
                #     print(f"Debugging...best frame for bucket: {best_bucket_frame_num}")
                #     print(
                #         "The result of is single color :",
                #         is_single_color(
                #             frame_dictionary_original[best_bucket_frame_num],
                #             maximum_robust=True,
                #         ),
                #     )
                #     cv2.imshow("Best frame", best_bucket_frame)
                #     key = cv2.waitKey(1)
                #     if key == ord("0"):
                #         break

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
    print(f"The total number of frames in the video, debug ({debug}): ", frame_number)
    # finding the final best frame to become the reference frame

    reference_frame_gray, reference_frame_num = best_frame_finder(
        final_frame_dictionary_gray, list(final_frame_dictionary_gray.keys())
    )

    reference_frame_original = final_frame_dictionary_original[reference_frame_num]

    print(f"Obtained best frame for video : {reference_frame_num}")

    del frame_number

    return reference_frame_original, reference_frame_gray, reference_frame_num


def test_reference_image_finder(video_path: str, sample_size: float = 0.5):
    """Validate if the selected frame from a given video is better than a sample of frames from that video.

    The sample can be small or it can be all the frames of the video.

    This function has its drawbacks due to the nature of choosing a best frame.

    It is simply the frame that is most similar to all frames, which won't
    necessarily extend to some subgroups of frames from the video.

    The metric of comparison is MSE, mean squared error."""

    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    # preserve the aspect ratio
    resize_factor = [0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    factor_found = False

    for factor in resize_factor:
        if factor_found:
            continue
        new_height = int(frame_height * factor)

        # if this condition does not trigger
        # then the resolution of the video
        # is substantially larger than 4K
        if 200 < new_height < 500:
            new_width = int(frame_width * factor)
            factor_found = True
            break
    print(
        "Finished finding the resize factor. Now using the reference image finder function..."
    )
    _, ref_gray, _ = reference_image_finder(
        video_path, buckets=fps, early_stop=False, resize_factor=(new_width, new_height)
    )
    print("Finished using the reference image finder function.")
    cap = cv2.VideoCapture(video_path)
    usable_frame_list = arr.array("i", [])
    frame_no = 0
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_no += 1

            if is_single_color(frame, maximum_robust=True):
                print("Frame is of one color. Skipping...")
                continue

            else:
                usable_frame_list.extend([frame_no])

        else:
            break

    cv2.destroyAllWindows()
    cap.release()

    usable_frame_total = len(usable_frame_list)
    test_frames_number = int(usable_frame_total * sample_size)
    test_frame_chosen = set(
        np.random.choice(usable_frame_list, size=test_frames_number, replace=False)
    )
    del usable_frame_list
    # test_frame_dictionary_original = dict()
    test_frame_dictionary_gray = dict()
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    print(
        f"Now testing reference frame against {sample_size*100}% of the video frames..."
    )
    print(f"Total number of usable frames in the video: {usable_frame_total}")
    print(f"Total number of frames chosen for testing: {test_frames_number}")

    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_number += 1

            if frame_number in test_frame_chosen:
                # test_frame_dictionary_original[frame_number] = frame.copy()
                test_frame_gray = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2GRAY)
                test_frame_dictionary_gray[frame_number] = test_frame_gray.copy()

            else:
                continue

        else:
            break

    cv2.destroyAllWindows()
    cap.release()

    del test_frame_chosen

    test_frame_dictionary_gray[-1] = ref_gray.copy()

    best_frame, best_frame_num = best_frame_finder(
        test_frame_dictionary_gray, list(test_frame_dictionary_gray.keys())
    )
    comparison = mse(best_frame, ref_gray)
    print(f"The MSE between the best frame and the reference frame is {comparison}")
    assert best_frame_num == -1, "The reference frame is not the best frame."
    assert mse(best_frame, ref_gray) == 0, "The reference frame is not the best frame."

    print("The reference frame is the best frame.")
