import os


class base_class:
    SKIP_FIRST_N_FRAMES = 60  # As some (most) videos start with grey screen
    RUN_FOR_FRAMES = 100  # Too low a value will cause a division by zero error
    DETECT_BOUNDING_SIZE = 50  # Size of the bounding box for detecition
    DRAW_BOUNDING_SIZE = 3  # Radius of the circle for the bounding box on the heatmap
    RESAMPLE = False  # Resample (from ns to ms) or choose the closest row


class aditya_config(base_class):
    ROOT_PATH = "/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/data"
    TEMP_OUTPUT_DIR = "." + os.sep + "output"


class eric_config(base_class):
    ROOT_PATH = r"C:\Users\ericr\Desktop\Data + Plus\eye tracking data from the museum in Rome (Pupil Invisible)"
    TEMP_OUTPUT_DIR = "." + os.sep + "output"


class april_config(base_class):
    ROOT_PATH = "/Users/aprilzuo/Downloads/eye tracking data from the museum in Rome (Pupil Invisible)"
    TEMP_OUTPUT_DIR = "." + os.sep + "output"
