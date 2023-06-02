# Bounding Box Experiments


# References : https://blog.roboflow.com/how-to-draw-a-bounding-box-in-python/


# Importing Libraries

import cv2
import pandas as pd
import datetime as dt

# path_0 = r"C:\Users\ericr\Desktop\Data + Plus\Tracking-Gazes-on-Museum-Pieces-Data-Plus\2022_30b\2022_30b\8bef8eba_0.0-63.584.mp4"
path_0 = r"C:\Users\ericr\Desktop\Data + Plus\Tracking-Gazes-on-Museum-Pieces-Data-Plus\2022_03bm\2022_03bm\2e6f4a06_0.0-65.563.mp4"
# path_0 = r"C:\Users\ericr\Desktop\Data + Plus\Tracking-Gazes-on-Museum-Pieces-Data-Plus\2022_39bm\2022_39bm\98b876d7_0.0-61.188.mp4"

# path_1 = r"C:\Users\ericr\Desktop\Data + Plus\Tracking-Gazes-on-Museum-Pieces-Data-Plus\2022_30b\2022_30b\gaze.csv"
path_1 = r"C:\Users\ericr\Desktop\Data + Plus\Tracking-Gazes-on-Museum-Pieces-Data-Plus\2022_03bm\2022_03bm\gaze.csv"
# path_1 = r"C:\Users\ericr\Desktop\Data + Plus\Tracking-Gazes-on-Museum-Pieces-Data-Plus\2022_39bm\2022_39bm\gaze.csv"

cap = cv2.VideoCapture(path_0)
gaze = pd.read_csv(path_1)

gaze["ts"] = gaze["timestamp [ns]"].apply(
    lambda x: dt.datetime.fromtimestamp(x / 1000000000)
)
baseline = gaze["ts"][0]
gaze["increment_marker"] = gaze["ts"] - baseline
gaze["seconds_id"] = gaze["increment_marker"].apply(lambda x: x.seconds) + 1
gaze_grouped_by_seconds = gaze.groupby("seconds_id")[
    ["gaze x [px]", "gaze y [px]"]
].mean()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # or 30
# fps = 30
fourcc = cv2.VideoWriter_fourcc(*"h264")  # mp4v
writer = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

i = 1
frame_counter = 0
while True:
    # `success` is a boolean and `frame` contains the next video frame
    success, frame = cap.read()
    print(f"the previous frame is {i}, success is {success}")
    if success:
        frame_counter += 1
        if frame_counter > 30:
            i += 1  # moving the group by one increment
            frame_counter = 1

        try:
            x = gaze_grouped_by_seconds.iloc[i, 0]
            y = gaze_grouped_by_seconds.iloc[i, 1]
            x = int(x)
            y = int(y)

        except:
            # leftover frames, use final one
            x = gaze_grouped_by_seconds.iloc[-1, 0]
            y = gaze_grouped_by_seconds.iloc[-1, 1]
            x = int(x)
            y = int(y)

        cv2.rectangle(frame, (x - 30, y - 30), (x + 30, y + 30), (0, 255, 0), 1)
        writer.write(frame)
        cv2.imshow("output", frame)
        print(f"passed {i}")
        i += 1
        if cv2.waitKey(1) & 0xFF == ord("s"):
            break

    else:
        break

    # wait 20 milliseconds between frames and break the loop if the `q` key is pressed
    # if cv2.waitKey(20) == ord('q'):
    #     break


# we also need to close the video and destroy all Windows
cv2.destroyAllWindows()
cap.release()
writer.release()
