"""This script describe statistics of when/where people are looking"""

import pandas as pd

# Records most frequently looked at/returned to feature
gaze_copy = pd.read_csv("gaze copy.csv")
mode = gaze_copy["tag"].mode()
print(mode)
