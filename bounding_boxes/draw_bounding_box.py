import cv2
import pandas as pd
import numpy as np
import datetime

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
video = cv2.VideoWriter('video.avi', fourcc, 30, (width, height))

frame_no = 1

while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    
    if frame_exists:
        current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        closet_value = min(gaze_csv['timestamp [ns]'], key=lambda x:abs(x-current_timestamp))
        closest_row = gaze_csv[gaze_csv['timestamp [ns]'] == closet_value].reset_index()
        x_pixel = round(closest_row['gaze x [px]'][0])
        y_pixel = round(closest_row['gaze x [px]'][0])
        size = 25
        curr_frame = cv2.rectangle(curr_frame, (x_pixel-size, y_pixel-size), (x_pixel+size, y_pixel+size), (0,0,255), 2)
        video.write(curr_frame)
    else:
        break
    frame_no += 1

cap.release()
video.release()