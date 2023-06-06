import cv2
import pandas as pd
import datetime as dt
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

# test image and test csv

gaze = pd.read_csv("gaze_fake_fix.csv")

gaze["ref_x"] = gaze["gaze x [px]"].apply(lambda x: np.random.randint(0, 1920))
gaze["ref_y"] = gaze["gaze y [px]"].apply(lambda x: np.random.randint(0, 1080))

img = cv2.imread("Fake Reference Image.png")

plt.imshow(img)
ls = []
i = 0
for coordinate in gaze.iterrows():
    x = coordinate[1]["ref_x"]
    y = coordinate[1]["ref_y"]
    img_copy = img.copy()
    cv2.rectangle(img, (x - 25, y - 25), (x + 25, y + 25), (255, 0, 0), 5)
    plt.imshow(img_copy)
    plt.show()
    ls.append(coordinate)

    wait = input("press enter to continue")

    if wait == "q":
        break

    i += 1
    if i == 10:
        break
