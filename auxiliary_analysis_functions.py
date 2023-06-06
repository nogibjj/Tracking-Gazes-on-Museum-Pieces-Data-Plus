"""This script holds auxiliary functions for gaze analysis,
to avoid bloating the analysis script."""


# fake tag creator
def fake_tagger(gaze_dataframe):
    """Create fake placeholder
    tags for analyses purposes,
    Will be defunct once real tags are created
    """
    import numpy as np

    np.random.seed(42)
    list_of_tags = ["eyes", "hands", "feet", "head", "torso", "background"]
    gaze_dataframe["tag"] = "something"
    gaze_dataframe["tag"] = gaze_dataframe["tag"].apply(
        lambda x: np.random.choice(list_of_tags)
    )
    return gaze_dataframe
