import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

cam = cv2.VideoCapture("./video_example.mp4")
flag = False
plot = plt.subplot(1, 1, 1)
plt.ion()


# working the first frame
ret, frame = cam.read()
if ret:
    histr = cv2.calcHist([frame], [0], None, [256], [0, 256])
    cv2.imshow('Frame', frame)
    # showing hist
    # plot = plot.imshow()

while (cam.isOpened):
    ret, frame = cam.read()
    if ret:
        # equalizing if flag is set
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # or convert
        if flag:
            frame = cv2.equalizeHist(frame)

        hist, bins = np.histogram(frame.flatten(), 256, [0, 256])

        # calculating hist
        histr = cv2.calcHist([frame], [0], None, [256], [0, 256])

        # showing frame
        cv2.imshow('Frame', frame)

        # showing hist
        # plot.set_data(plt.hist(histr))

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('h'):
            flag = not flag
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    # Break the loop
    else:
        break

    # When everything done, release
    # the video capture object
cam.release()
plt.ioff()

# Closes all the frames
cv2.destroyAllWindows()
