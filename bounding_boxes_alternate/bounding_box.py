# Bounding Box Experiments


# References : https://blog.roboflow.com/how-to-draw-a-bounding-box-in-python/


# Importing Libraries

import cv2
import pandas as pd
import datetime as dt

cap = cv2.VideoCapture(
    r"C:\Users\ericr\Desktop\Data + Plus\Tracking-Gazes-on-Museum-Pieces-Data-Plus\2022_30b\2022_30b\8bef8eba_0.0-63.584.mp4"
)


gaze = pd.read_csv(
    r"C:\Users\ericr\Desktop\Data + Plus\Tracking-Gazes-on-Museum-Pieces-Data-Plus\2022_30b\2022_30b\gaze.csv"
)

gaze["ts"] = gaze["timestamp [ns]"].apply(lambda x: dt.datetime.utcfromtimestamp(x))

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# fps = int(cap.get(cv2.CAP_PROP_FPS))
fps = int(cap.get(cv2.CAP_PROP_FPS))  # or 30
# fps = 30
fourcc = cv2.VideoWriter_fourcc(*"h264")  # mp4v
writer = cv2.VideoWriter("output.mp4", fourcc, fps, (frame_width, frame_height))

i = 0

while True:
    # `success` is a boolean and `frame` contains the next video frame
    success, frame = cap.read()
    print(f"the previous frame is {i}, success is {success}")
    if success:
        x = 782.720
        y = 486.945
        x = int(x)
        y = int(y)
        cv2.rectangle(frame, (x - 20, y - 20), (x + 20, y + 20), (0, 255, 0), 1)
        # cv2.rectangle(frame, (100, 100), (500, 500), (0, 255, 0), -1)
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
