### Adding temporary solution for multiple folders
import os
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import datetime as dt
import sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, path)

from config import *
from heatmap.functions import create_directory

# # print(" Made it past imports")
# # Set env variables based on config file
# try:
#     env = sys.argv[1]
#     env_var = eval(env + "_config")
# except:
#     print("Enter valid env variable. Refer to classes in the config.py file")
#     sys.exit()

# # to receive a copy of the tag coordinates
# create_directory(env_var.TEMP_OUTPUT_DIR)

# ROOT_PATH = env_var.ROOT_PATH
ROOT_PATH = r"C:\Users\ericr\Desktop\Data + Plus\eye tracking data from the museum in Rome (Pupil Invisible)"
# MEMBER_FLAG = env

participant_paths_folders = []

for folder in os.listdir(ROOT_PATH):
    try:
        new_path = os.path.join(ROOT_PATH, folder)
        os.listdir(new_path)
        print(f"Running for folder -- {folder}")
        print(f"{folder} is a directory")
        participant_paths_folders.append(new_path)

    except:
        print(f"{folder} is a file")
        continue

checkpoint = False
last_folder = None

participant_list = []

participant_paths_folders = sorted(participant_paths_folders)

# fixing news issue - Warning - Temporary fix
for folder in participant_paths_folders:
    files = os.listdir(folder)
    participant_id = folder.split(os.sep)[-1]
    if "new" in participant_id:
        print(f"Fixing participant id -- {participant_id}")
        temp = participant_id.replace("new", "")
        flip_flag = False
        for id in participant_list:
            if temp in id:
                print(f"Replacing {id} with {participant_id}")
                ind = participant_list.index(id)
                flip_flag = True
                participant_list.pop(ind)
                participant_list.append(participant_id)
        if not (flip_flag):
            participant_list.append(participant_id)
            flip_flag = False
    else:
        participant_list.append(participant_id)

participant_list = sorted(participant_list)

if checkpoint:
    with open("tagger_last_folder.txt", "r") as f:
        last_folder = f.read()

    index_change = participant_paths_folders.index(last_folder)

    participant_paths_folders = participant_paths_folders[index_change + 1 :]

gazes_paths = []
gazes_reference_images = []
gazes_participants = []
full_gaze = None
for folder in participant_paths_folders:
    files = os.listdir(folder)
    print(folder)
    participant_id = folder.split(os.sep)[-1]
    for file in files:
        if "gaze.csv" in file:
            gazes_paths.append(os.path.join(folder, file))
            gazes_participants.append(participant_id)
            if full_gaze is None:
                full_gaze = pd.read_csv(os.path.join(folder, file))

            else:
                full_gaze = pd.concat(
                    [full_gaze, pd.read_csv(os.path.join(folder, file))],
                    ignore_index=True,
                )

        if "reference_image" in file:
            gazes_reference_images.append(os.path.join(folder, file))

n = 14
trial_file = pd.read_csv(gazes_paths[n])
trial_participant = gazes_participants[n]
trial_reference_image = gazes_reference_images[n]


base_img = cv2.imread(trial_reference_image)


img = base_img
reset_img = img.copy()

plt.imshow(img)


import pandas as pd
import numpy as np
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.metrics import v_measure_score


reduced_df = trial_file[["gaze x [px]", "gaze y [px]"]].copy()
reduced_df = reduced_df.rename(columns={"gaze x [px]": "x", "gaze y [px]": "y"})
full_gaze_essential = full_gaze[full_gaze["fixation id"].notnull()].copy()
full_gaze_essential = full_gaze[["gaze x [px]", "gaze y [px]"]].copy()
full_gaze_essential = full_gaze_essential.rename(
    columns={"gaze x [px]": "x", "gaze y [px]": "y"}
)
full_gaze_essential = full_gaze_essential.reset_index(drop=True)
# overlay the gaze points on the image
for i in range(len(full_gaze_essential)):
    x = int(full_gaze_essential.iloc[i, 0])
    y = int(full_gaze_essential.iloc[i, 1])
    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

plt.imshow(img)


dbscan_cluster1 = DBSCAN(eps=10, min_samples=10)
dbscan_cluster1.fit(reduced_df)

# Visualizing DBSCAN
plt.scatter(
    reduced_df.iloc[:, 0],
    reduced_df.iloc[:, 1],
    c=dbscan_cluster1.labels_,
    cmap="rainbow",
    label="Points in cluster",
)
plt.xlabel("$x_1$")
plt.ylabel("$x_2$")

# a list of colors for the clusters
colors = ["red", "green", "blue", "yellow", "orange", "purple", "pink", "gray"]

# overlay the colored gaze points on the image
for i in range(len(reduced_df)):
    x = int(reduced_df.iloc[i, 0])
    y = int(reduced_df.iloc[i, 1])
    cv2.circle(img, (x, y), 1, (0, 0, 255), -1)


# Number of Clusters
labels = dbscan_cluster1.labels_
N_clus = len(set(labels)) - (1 if -1 in labels else 0)
print("Estimated no. of clusters: %d" % N_clus)

# Identify Noise
n_noise = list(dbscan_cluster1.labels_).count(-1)
print("Estimated no. of noise points: %d" % n_noise)

# Calculating v_measure
# print('v_measure =', v_measure_score(y, labels))

# Calculating silhouette_score
# print('silhouette_score =', silhouette_score(reduced_df, labels))


dummy_dg = pd.DataFrame(columns=["x", "y"])
dummy_dg["x"] = dbscan_cluster1.components_[:, 0]
dummy_dg["y"] = dbscan_cluster1.components_[:, 1]
