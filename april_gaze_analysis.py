"""This script describe statistics of when/where people are looking"""

import pandas as pd
import datetime as dt
from auxiliary_analysis_functions import fake_tagger

# Records most frequently looked at/returned to feature
gaze_copy = pd.read_csv("gaze_fake_fix.csv")


gaze_copy = fake_tagger(gaze_copy)
mode = gaze_copy["tag"].mode()
print(mode)

gaze_copy["ts"] = gaze_copy["timestamp [ns]"].apply(
    lambda x: dt.datetime.fromtimestamp(x / 1000000000)
)
baseline = gaze_copy["ts"][0]
gaze_copy["increment_marker"] = gaze_copy["ts"] - baseline
gaze_copy["seconds_id"] = gaze_copy["increment_marker"].apply(lambda x: x.seconds) + 1
