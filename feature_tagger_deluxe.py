### Adding temporary solution for multiple folders
from repository_finder import repository_details
import os

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
    drawing = True

    feature_coordinates = []
