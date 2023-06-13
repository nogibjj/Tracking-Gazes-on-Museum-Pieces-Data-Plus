"""
Author: Aditya John (aj391)
This is a script that ingests a csv of gazes (when looking at an object) and outputs a final image with a heatmap across the various pixels of interest. 
It also updates the gaze.csv file with new 'gaze' pixel locations that correspond to the reference image instead of the actual video. 

# ToDo:
- Add smoothing to the video file [pixel closeness? frame combination?]
- Figure out how to skip the first few grey frames in the video [skip frame if % of greyness > thresh?]
- Change opacity of the bounding boxes? 
Ref: https://stackoverflow.com/questions/56472024/how-to-change-the-opacity-of-boxes-cv2-rectangle
- QC the outputs - April?
"""
import sys
from collections import defaultdict
import pandas as pd
import cv2
from functions import convert_timestamp_ns_to_ms, get_closest_individual_gaze_object, get_closest_reference_pixel, normalize_heatmap_dict, draw_heatmap_on_ref_img, create_output_directory


for folder in folder_files:
    print(f'Running for folder -- {folder}')
    pixel_heatmap = defaultdict(int)
    frame_no = 0folder_files = ['2022_03bm/2022_03bm', '2022_30b/2022_30b', '2022_39bm/2022_39bm']
    #ToDo: Write a function to skip the first n frames based on percentage of grey/black color in the image
    skip_first_n_frames = 60 # as the video starts with grey 
    run_for_frames = 100 # Too low a value will cause a division by zero error
    bounding_size = 25
    name = folder.split('/')[0]
    csv_file = f'/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/{folder}/fixations.csv'
    video_file = f'/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/{folder}/video.mp4'
    gaze_csv = pd.read_csv(csv_file)
    #gaze_csv = convert_timestamp_ns_to_ms(gaze_csv)
    gaze_csv = convert_timestamp_ns_to_ms(gaze_csv, 'start timestamp [ns]')
    gaze_csv = convert_timestamp_ns_to_ms(gaze_csv, 'end timestamp [ns]')
    gaze_csv.to_csv('updated_fixation.csv', index=False)

    updated_gaze = pd.DataFrame()
    cap = cv2.VideoCapture(video_file)

    create_output_directory()
    while(cap.isOpened()):
        if frame_no % 1000 == 0:
            print(f'Processed {frame_no} frames')
        frame_no += 1
        frame_exists, curr_frame = cap.read()
        current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        print(current_timestamp)
        continue
        if frame_no < skip_first_n_frames:
            continue

        ##### Uncomment below if early stopping is required
        # if frame_no > skip_first_n_frames + run_for_frames:
        #    break
        
        elif frame_no == skip_first_n_frames and frame_exists:
            first_frame = curr_frame
            cv2.imwrite(f'output/reference_image_{name}.png', first_frame)

        elif frame_exists:
            gaze_object_crop, closest_row = get_closest_individual_gaze_object(cap, curr_frame, gaze_csv, bounding_size)
            closest_ref_pixel = get_closest_reference_pixel(first_frame, gaze_object_crop)
            closest_row['ref_x_pixel'] = closest_ref_pixel[0]
            closest_row['ref_y_pixel'] = closest_ref_pixel[1]
            updated_gaze = pd.concat([updated_gaze, closest_row])
            pixel_heatmap[closest_ref_pixel] += 1

        else:
            break
    print(pixel_heatmap[0, 0])
    normalized_heatmap_dict = normalize_heatmap_dict(pixel_heatmap)
    final_img = draw_heatmap_on_ref_img(pixel_heatmap, first_frame, bounding_size)

    cv2.imwrite(f'output/heatmap_output_{name}.png', final_img)
    updated_gaze.to_csv(f'output/updated_gaze_{name}.csv', index=False)
    cap.release()
