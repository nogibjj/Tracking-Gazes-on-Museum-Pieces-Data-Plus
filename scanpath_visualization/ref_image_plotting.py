import cv2
import pandas as pd
import numpy as np
import sys
import os
import glob
import traceback

# prepend parent directory to the system path:
path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from config import *

try:
    env = sys.argv[1]
    env_var = eval(env + "_config")
except Exception as ee:
    print("Enter valid env variable. Refer to classes in the config.py file")
    sys.exit()

ref_image = cv2.imread('reference_image.png')
for file in ['sift_2.csv', 'SIFT.csv', 'template_2.csv', 'template.csv']: # for index, folder in enumerate(os.listdir(env_var.ROOT_PATH)):
    try:
        print(f'Running for file {file}')
        gaze_csv = pd.read_csv(file)

        gaze_csv['timestamp [ns]'] = pd.to_datetime(gaze_csv['timestamp [ns]'])
        start_timestamp = gaze_csv['timestamp [ns]'][0]
        gaze_csv['timestamp [ns]'] = (gaze_csv['timestamp [ns]'] - start_timestamp)
        gaze_csv['timestamp [ns]'] = gaze_csv['timestamp [ns]'].astype(np.int64) / int(1e6)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
        name = file.split('.')[0]
        video = cv2.VideoWriter(f'outputs/ref_image/scanpath_ref_image_{name}.mp4', fourcc, 30, (ref_image.shape[0], ref_image.shape[1]))

        queue = []
        keep_last = 15

        for index, row in gaze_csv.iterrows():
                x_pixel = round(row['ref_center_x'])
                y_pixel = round(row['ref_center_y'])

                queue.append((x_pixel, y_pixel))
        
                if len(queue) > keep_last:
                    ref_image = cv2.imread('reference_image.png')
                    ref_image = cv2.circle(ref_image, queue[-1], 15, (0,0,255), 2)
                    for idx, point in enumerate(queue):
                        if idx !=0 :
                            ref_image = cv2.line(ref_image, point, queue[idx-1], (255, 255, 255), 3)

                    queue.pop(0) 
                    if index % 500 == 0:
                        cv2.imwrite(f'outputs/ref_image/ref_{name}_{index}.png', ref_image)

                    video.write(ref_image)     
        video.release()
    except Exception as ee:
         print(ee)