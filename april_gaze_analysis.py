"""This script describe statistics of when/where people are looking"""

import pandas as pd
import datetime as dt

# Records most frequently looked at/returned to feature
gaze_copy = pd.read_csv("gaze_fake_fix.csv")


# fake tag creator
def fake_tagger(gaze_dataframe):
    """Create fake placeholder
    tags for analyses purposes
    """
    import random
    import numpy as np

    np.random.seed(42)
    list_of_tags = ["eyes", "hands", "feet", "head", "torso", "background"]
    gaze_dataframe["tag"] = "something"
    gaze_dataframe["tag"] = gaze_dataframe["tag"].apply(
        lambda x: np.random.choice(list_of_tags)
    )
    assert gaze_copy["tag"].value_counts()[0] == 2171
    return gaze_dataframe


gaze_copy = fake_tagger(gaze_copy)
mode = gaze_copy["tag"].mode()
print(mode)

gaze_copy["ts"] = gaze_copy["timestamp [ns]"].apply(
    lambda x: dt.datetime.fromtimestamp(x / 1000000000)
)
baseline = gaze_copy["ts"][0]
gaze_copy["increment_marker"] = gaze_copy["ts"] - baseline
gaze_copy["seconds_id"] = gaze_copy["increment_marker"].apply(lambda x: x.seconds) + 1
