### Adding temporary solution for multiple folders
from repository_finder import repository_details
import os
from tag_event_functions import drawfunction
import cv2

ROOT_PATH, ART_PIECE = repository_details("Paths.txt")

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

for folder in participant_paths_folders:
    files = os.listdir(folder)

    for single_file in files:
        if ".mp4" in single_file:
            file = os.path.join(folder, single_file)
            print(f"Running for file -- {single_file}")
    print(f"file is {file}")
    base_img = cv2.imread(file)
    # base_img = cv2.imread("test4 image prompter.jpg")
    # img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
    img = base_img
    reset_img = img.copy()
    plt.imshow(img)
