import cv2
import pandas as pd
import numpy as np
import datetime
import numpy as np
from matplotlib import pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import normalize
from pprint import pprint
import matplotlib
from matplotlib import cm

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

# fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
# fps = 1
# video = cv2.VideoWriter('video.avi', fourcc, fps, (width, height))

frame_no = 0
skip_first_n_frames = 25
run_for_frames = 1500
first_frame = ""
size = 25
pixel_heatmap = defaultdict(int)

# def get_cropped_gaze_img():

while(cap.isOpened()):
    frame_no += 1
    frame_exists, curr_frame = cap.read()
    if frame_no < skip_first_n_frames:
        continue

    if frame_no > skip_first_n_frames + run_for_frames:
        break
    
    if frame_no == skip_first_n_frames and frame_exists:
        first_frame = curr_frame

    if frame_exists:

        current_timestamp = cap.get(cv2.CAP_PROP_POS_MSEC)
        closet_value = min(gaze_csv['timestamp [ns]'], key=lambda x:abs(x-current_timestamp))
        closest_row = gaze_csv[gaze_csv['timestamp [ns]'] == closet_value].reset_index()
        x_pixel = round(closest_row['gaze x [px]'][0])
        y_pixel = round(closest_row['gaze x [px]'][0])
        template = curr_frame[y_pixel:y_pixel+size, x_pixel:x_pixel+size]

    else:
        break

    frame_no += 1
    
    # template.shape[0] is 3 (because the picture is RGB)
    
    chosen_method = 1
    methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv.TM_CCORR',
            'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

    method = eval(methods[chosen_method])
    res = cv2.matchTemplate(first_frame, template, method)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
    # if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
    #     top_left = min_loc
    # else:
    #     top_left = max_loc
    top_left = max_loc
    pixel_heatmap[top_left] += 1

"""
Normalize heatmap
"""
pprint(pixel_heatmap)
values = list(pixel_heatmap.values())
norm = matplotlib.colors.Normalize(vmin=min(values), vmax=max(values), clip=True)
mapper = cm.ScalarMappable(norm=norm, cmap=cm.Greys_r)
rgba_colors = mapper.to_rgba(values)
pixel_heatmap = {key: tuple(np.array(color[:3]) * 255) for key, color in zip(pixel_heatmap.keys(), rgba_colors)}
pprint(pixel_heatmap)

for key, value in pixel_heatmap.items():

    bottom_right = (key[0] + size, key[1] + size)
    op_frame = cv2.rectangle(first_frame, key, bottom_right, value, 2)


cv2.imwrite('output.png', op_frame)
cap.release()
# video.release()

