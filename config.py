import os


class base_class:
    SKIP_FIRST_N_FRAMES = 60  # As some (most) videos start with grey screen
    RUN_FOR_FRAMES = 100  # Too low a value will cause a division by zero error
    DETECT_BOUNDING_SIZE = 50  # Size of the bounding box for detecition
    DRAW_BOUNDING_SIZE = 3  # Radius of the circle for the bounding box on the heatmap
    RESAMPLE = False  # Resample (from ns to ms) or choose the closest row
    REFERENCE_IMAGE = True

    # PLACE THE NAME OF YOUR ART PIECE AFTER THE ART PIECE VARIABLE
    # IF THERE IS MORE THAN ONE ART PIECE OR OBJECT OF INTEREST IN THE VIDEO
    # TURN THE OBJECT OF INTEREST INTO A LIST LIKE ["ART PIECE 1", "ART PIECE 2"]
    ART_PIECE = ["Truscan Couple Statue"]


class aditya_config(base_class):
    ROOT_PATH = "/workspaces/Tracking-Gazes-on-Museum-Pieces-Data-Plus/data"
    TEMP_OUTPUT_DIR = "." + os.sep + "output"


class eric_config(base_class):
    # ROOT_PATH = r"C:\Users\ericr\Desktop\Data + Plus\eye tracking data from the museum in Rome (Pupil Invisible)"
    ROOT_PATH = r"C:\Users\ericr\Desktop\Data + Plus\VulciFinal"
    TEMP_OUTPUT_DIR = "." + os.sep + "output"


class april_config(base_class):
    # ROOT_PATH = "/Users/aprilzuo/Downloads/eye tracking data from the museum in Rome (Pupil Invisible)"
    ROOT_PATH = "/Users/aprilzuo/Desktop/Duke/Research/Data+/VulciFinal"
    TEMP_OUTPUT_DIR = "." + os.sep + "output"


# INSTRUCTIONS

# These instructions assume you have already downloaded the data
# and divided it into the folders of your participants
# In addition, we expect the folders to contain a csv file with the gaze coordinates and a video mp4 file.


class user_config(base_class):
    # Place your path with all of your participant folders after the ROOT_PATH variable
    # Erase "None" and put your path in quotes like this r"PATH"
    ROOT_PATH = None
    #  C:\Users\ericr\Desktop\Data + Plus\eye tracking data from the museum in Rome (Pupil Invisible)
    TEMP_OUTPUT_DIR = "." + os.sep + "output"
