import cv2
import numpy as np

test_path = r"C:\Users\ericr\Downloads\cat video test for ref image finder.mp4"


def mse(img1, img2):
    h, w = img1.shape
    diff = cv2.subtract(img1, img2)
    err = np.sum(diff**2)
    mse = err / (float(h * w))
    return mse


img1 = cv2.imread(
    r"C:\Users\ericr\Desktop\Data + Plus\Tracking-Gazes-on-Museum-Pieces-Data-Plus\test3 image prompter.jpg"
)
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


cap = cv2.VideoCapture(test_path)
frame_dictionary = dict()
frame_number = 0
while cap.isOpened():
    success, frame = cap.read()

    if success:
        frame_number += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_dictionary[frame_number] = frame

    else:
        print("Done reading the video")  # add more to print statement
        print("Last frame number: ", frame_number)
        break

cv2.destroyAllWindows()
cap.release()
frame_mse = dict()
frame_mse_with_list = dict()

for main_key in frame_dictionary:
    copy_dn = frame_dictionary.copy()
    copy_dn.pop(main_key)
    mse_list = list()

    for other_key in copy_dn:
        mse_list.extend([mse(frame_dictionary[main_key], copy_dn[other_key])])

    # Mean is the best option.
    # Mode won't work because the participant is never still.
    # Median is not necessary because it is a metric meant to
    # protect against outliers.
    # However, in a video with 1000 frames and 20 jittery/uncommon ones,
    # those 20 bad ones will be outliers in all the MSEs of the
    # good frames, but they will simultaneously have the worst MSE.
    frame_mse[main_key] = np.mean(mse_list.copy())
    frame_mse_with_list[main_key] = (np.mean(mse_list), mse_list.copy())

best_frame = min(frame_mse, key=frame_mse.get)


############################################3
# Function version


def reference_image_finder(video_path: str, return_mse_list=False):
    import cv2
    import numpy as np

    # mse function for the image calculations
    def mse(img1, img2):
        h, w = img1.shape
        diff = cv2.subtract(img1, img2)
        err = np.sum(diff**2)
        mse = err / (float(h * w))
        return mse

    cap = cv2.VideoCapture(video_path)
    frame_dictionary = dict()
    frame_number = 0
    while cap.isOpened():
        success, frame = cap.read()

        if success:
            frame_number += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_dictionary[frame_number] = frame

        else:
            print("Done storing the video frames")  # add more to print statement
            print(
                f"Last frame number for {video_path.split(os.sep)[-1]} : {frame_number}"
            )
            break

    cv2.destroyAllWindows()
    cap.release()
    frame_mse = dict()
    frame_mse_with_list = dict()

    for main_key in frame_dictionary:
        copy_dn = frame_dictionary.copy()
        copy_dn.pop(main_key)
        mse_list = list()

        for other_key in copy_dn:
            mse_list.extend([mse(frame_dictionary[main_key], copy_dn[other_key])])

        # Mean is the best option.
        # Mode won't work because the participant is never still.
        # Median is not necessary because it is a metric meant to
        # protect against outliers.
        # However, in a video with 1000 frames and 20 jittery/uncommon ones,
        # those 20 bad ones will be outliers in all the MSEs of the
        # good frames, but they will simultaneously have the worst MSE.
        frame_mse[main_key] = np.mean(mse_list.copy())
        frame_mse_with_list[main_key] = (np.mean(mse_list), mse_list.copy())

    best_frame = min(frame_mse, key=frame_mse.get)
