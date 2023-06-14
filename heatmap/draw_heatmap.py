"""
Author: Aditya John (aj391)
This is a script that ingests a csv of gazes (when looking at an object) and outputs a final image 
with a heatmap across the various pixels of interest. 
It also updates the gaze.csv file with new 'gaze' pixel locations that correspond to the reference image 
instead of the actual video. 

# ToDo:
- Add smoothing to the video file [pixel closeness? frame combination?]
- Figure out how to skip the first few grey frames in the video [skip frame if % of greyness > thresh?]
- Change opacity of the bounding boxes? 
Ref: https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
- QC the outputs - April?
"""
from collections import defaultdict
import pandas as pd
import cv2
from functions import (
    convert_timestamp_ns_to_ms,
    get_closest_individual_gaze_object,
    get_closest_reference_pixel,
    normalize_heatmap_dict,
    draw_heatmap_on_ref_img,
)
import os
import glob

# root_path = "/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/"
# root_path = r"C:\Users\ericr\Desktop\Data + Plus\eye tracking data from the museum in Rome (Pupil Invisible)"
folder_files = ["2022_03bm/2022_03bm", "2022_30b/2022_30b", "2022_39bm/2022_39bm"]

root_path = r"C:\Users\ericr\Desktop\Data + Plus\eye tracking data from the museum in Rome (Pupil Invisible)"


# for folder in folder_files:
for folder in os.listdir(root_path):
    folder = os.path.join(root_path, folder)
    print(f"Running for folder -- {folder}")
    pixel_heatmap = defaultdict(int)
    frame_no = 0

    # ToDo: Write a function to skip the first n frames based on percentage of grey/black color in the image
    skip_first_n_frames = 60  # as the video starts with grey
    run_for_frames = 100  # Too low a value will cause a division by zero error
    bounding_size = 250
    name = folder.split("\\")[-1]
    print(f"The name of the participant is : {name}")
    # csv_file = f"{root_path}/{folder}/gaze.csv"
    csv_file = os.path.join(folder, "gaze.csv")
    # video_file = f"{root_path}/{folder}/video.mp4"
    video_file = os.path.join(folder, "*.mp4")
    video_file = glob.glob(video_file)[0]
    # print(glob.glob(video_file))
    gaze_csv = pd.read_csv(csv_file)
    gaze_csv = convert_timestamp_ns_to_ms(gaze_csv)

    updated_gaze = pd.DataFrame()
    cap = cv2.VideoCapture(video_file)

    while cap.isOpened():
        if frame_no % 1000 == 0:
            print(f"Processed {frame_no} frames")
        frame_no += 1
        frame_exists, curr_frame = cap.read()

        if frame_no < skip_first_n_frames:
            continue

        ##### Uncomment below if early stopping is required
        # if frame_no > skip_first_n_frames + run_for_frames:
        #     break

        elif frame_no == skip_first_n_frames and frame_exists:
            first_frame = curr_frame
            # cv2.imwrite(f"{root_path}/{folder}/reference_image_{name}.png", first_frame)
            cv2.imwrite(f"output_heatmap/{name}_reference_image.png", first_frame)

        elif frame_exists:
            gaze_object_crop, closest_row = get_closest_individual_gaze_object(
                cap, curr_frame, gaze_csv, bounding_size
            )
            ref_center = get_closest_reference_pixel(first_frame, gaze_object_crop)
            closest_row["ref_center_x"] = ref_center[0]
            closest_row["ref_center_y"] = ref_center[1]
            updated_gaze = pd.concat([updated_gaze, closest_row])
            pixel_heatmap[ref_center] += 1

        else:
            break

    normalized_heatmap_dict = normalize_heatmap_dict(pixel_heatmap)
    final_img = draw_heatmap_on_ref_img(pixel_heatmap, first_frame, bounding_size)

    # cv2.imwrite(
    #    f"{root_path}/{folder}/heatmap_output_{name}_{bounding_size}.png", final_img
    # )

    cv2.imwrite(f"output_heatmap/{name}_heatmap.png", final_img)

    # updated_gaze.to_csv(f"{root_path}/{folder}/updated_gaze_{name}.csv", index=False)
    cap.release()
