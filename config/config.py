import os


class base_class:
    SKIP_FIRST_N_FRAMES = 60  # As some (most) videos start with grey screen
    RUN_FOR_FRAMES = 100  # Too low a value will cause a division by zero error
    DETECT_BOUNDING_SIZE = 50  # Size of the bounding box for detecition
    DRAW_BOUNDING_SIZE = 3  # Radius of the circle for the bounding box on the heatmap
    RESAMPLE = False  # Resample (from ns to ms) or choose the closest row
    REFERENCE_IMAGE = True  # Flag indicating if reference image is manually found
    ART_PIECE = "vulci_site"  # Name of the art piece/folder where the data is kept
    ART_PIECE = "truscan_statue"  # Name of the art piece/folder where the data is kept


class aditya_config(base_class):
    ROOT_PATH = "/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/data"
    OUTPUT_PATH = "/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/output"


class eric_config(base_class):
    ROOT_PATH = r"C:\Users\ericr\Desktop\Data + Plus\eye tracking data from the museum in Rome (Pupil Invisible)"
    # ROOT_PATH = r"C:\Users\ericr\Desktop\Data + Plus\eye tracking data from the museum in Rome (Pupil Invisible)"
    OUTPUT_PATH = r"C:\Users\ericr\Desktop\Data + Plus\output"
    REFERENCE_IMAGE = False  # Flag indicating if reference image is manually found


class april_config(base_class):
    ROOT_PATH = "/Users/aprilzuo/Downloads/eye tracking data from the museum in Rome (Pupil Invisible)"
    OUTPUT_PATH = "/Users/aprilzuo/Downloads/eye tracking data from the museum in Rome (Pupil Invisible)/output"
