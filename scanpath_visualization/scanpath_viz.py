import cv2
import pandas as pd
import numpy as np
import datetime

csv_file = '/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/scanpath_visualization/gaze.csv'
video_file = '/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/scanpath_visualization/28f843bc_0.0-60.632.mp4'
gaze_csv = pd.read_csv(csv_file)

gaze_csv['timestamp [ns]'] = pd.to_datetime(gaze_csv['timestamp [ns]'])
start_timestamp = gaze_csv['timestamp [ns]'][0]
gaze_csv['timestamp [ns]'] = (gaze_csv['timestamp [ns]'] - start_timestamp)
gaze_csv['timestamp [ns]'] = gaze_csv['timestamp [ns]'].astype(np.int64) / int(1e6)

rolling_window = 30
gaze_csv['x_smooth'] = gaze_csv['gaze x [px]'].rolling(rolling_window).mean()
gaze_csv['y_smooth'] = gaze_csv['gaze y [px]'].rolling(rolling_window).mean()
gaze_csv = gaze_csv[rolling_window:]
fixation_centers = gaze_csv[['x_smooth', 'y_smooth', 'fixation id']].groupby('fixation id').mean().reset_index()

cap = cv2.VideoCapture(video_file)
frame_exists, curr_frame = cap.read()
height,width,layers=curr_frame.shape

fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
video = cv2.VideoWriter('video.avi', fourcc, 30, (width, height))

frame_no = 1
prev_point = None
queue = []
keep_last = 100


while(cap.isOpened()):
    frame_exists, curr_frame = cap.read()
    
    if frame_exists:
        current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        closet_value = min(gaze_csv['timestamp [ns]'], key=lambda x:abs(x-current_timestamp))
        closest_row = gaze_csv[gaze_csv['timestamp [ns]'] == closet_value].reset_index()
        x_pixel = round(closest_row['x_smooth'][0])
        y_pixel = round(closest_row['y_smooth'][0])
        fixation_id = closest_row['fixation id'][0]
        if fixation_id == fixation_id:
            fixation_center = fixation_centers[fixation_centers['fixation id'] == fixation_id].reset_index()
            curr_centre_x = round(fixation_center['x_smooth'][0])
            curr_centre_y = round(fixation_center['y_smooth'][0])
            # curr_frame = cv2.circle(curr_frame, (curr_centre_x, curr_centre_y), 15, (0,0,255), 2)

            queue.append((x_pixel, y_pixel))
            if fixation_id > 1:
                prev_fixation_center = fixation_centers[fixation_centers['fixation id'] == (fixation_id - 1)].reset_index()
                prev_centre_x = round(prev_fixation_center['x_smooth'][0])
                prev_centre_y = round(prev_fixation_center['y_smooth'][0])

                # curr_frame = cv2.line(curr_frame, (curr_centre_x, curr_centre_y), (prev_centre_x, prev_centre_y), (255, 255, 255), 3)
        
        # if prev_point:
        #     curr_frame = cv2.line(curr_frame, prev_point, (x_pixel, y_pixel), (255, 255, 255), 5) 

        # prev_point = (x_pixel, y_pixel)

        # curr_frame = cv2.circle(curr_frame, (x_pixel, y_pixel), 15, (0,0,255), 2)

        
        if len(queue) > keep_last:
            curr_frame = cv2.circle(curr_frame, queue[-1], 15, (0,0,255), 2)
            for idx, point in enumerate(queue):
                if idx !=0 :
                    curr_frame = cv2.line(curr_frame, point, queue[idx-1], (255, 255, 255), 3)

            queue.pop(0) 
        video.write(curr_frame)
        
    else:
        break


    frame_no += 1

cap.release()
video.release()