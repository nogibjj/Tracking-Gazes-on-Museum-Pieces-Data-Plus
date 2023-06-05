import cv2
import pandas as pd
import numpy as np
import datetime
import numpy as np
from matplotlib import pyplot as plt

# Todo: Add loop to this
csv_file = '/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/2022_30b/2022_30b/gaze.csv'
video_file = '/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/2022_30b/2022_30b/8bef8eba_0.0-63.584.mp4'
gaze_csv = pd.read_csv(csv_file)

gaze_csv['timestamp [ns]'] = pd.to_datetime(gaze_csv['timestamp [ns]'])
start_timestamp = gaze_csv['timestamp [ns]'][0]
gaze_csv['timestamp [ns]'] = (gaze_csv['timestamp [ns]'] - start_timestamp)
gaze_csv['timestamp [ns]'] = gaze_csv['timestamp [ns]'].astype(np.int64) / int(1e6)

cap = cv2.VideoCapture(video_file)
frame_exists, curr_frame = cap.read()
height,width,layers=curr_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
fps = 1
video = cv2.VideoWriter('video.avi', fourcc, fps, (width, height))

frame_no = 1

first_frame = ""
while(cap.isOpened()):

    if frame_no > 10:
        break
    
    frame_exists, curr_frame = cap.read()

    if frame_no == 1 and frame_exists:
        first_frame = curr_frame
    
    if frame_exists:
        current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        closet_value = min(gaze_csv['timestamp [ns]'], key=lambda x:abs(x-current_timestamp))
        closest_row = gaze_csv[gaze_csv['timestamp [ns]'] == closet_value].reset_index()
        x_pixel = round(closest_row['gaze x [px]'][0])
        y_pixel = round(closest_row['gaze x [px]'][0])
        size = 25
        template = curr_frame[y_pixel:y_pixel+size, x_pixel:x_pixel+size]
        # curr_frame = cv2.resize(curr_frame, (1080, 1920), interpolation = cv2.INTER_AREA)
        # Below is to add bounding box
        curr_frame = cv2.rectangle(curr_frame, (x_pixel-size, y_pixel-size), (x_pixel+size, y_pixel+size), (0,0,255), 2)
        

    else:
        break
    frame_no += 1
    # template.shape[0] is 3 (because the picture is RGB)
    w, h = template.shape[1], template.shape[2] 
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']
    methods = ['cv2.TM_CCOEFF']
    for meth in methods:
        method = eval(meth)
        res = cv2.matchTemplate(first_frame,template,method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    curr_frame = cv2.rectangle(curr_frame, top_left, bottom_right, (0,255,0), 2)
    video.write(curr_frame)

cap.release()
video.release()

