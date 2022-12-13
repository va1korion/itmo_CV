import cv2
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter


cam = cv2.VideoCapture("./examples/video_example.mp4")
flag = False
plot = plt.subplot(1, 1, 1)
plt.ion()


def dark_magic(histr):
    hist_w = 512
    hist_h = 400
    bin_w = int(round(hist_w / 256))
    histImage = np.zeros((512, 400), dtype=np.uint8)
    cv2.normalize(histr, histr, alpha=0, beta=512, norm_type=cv2.NORM_MINMAX)
    for i in range(1, 256):
        cv2.line(histImage, (bin_w * (i - 1), hist_h - int(histr[i - 1])),
                 (bin_w * (i), hist_h - int(histr[i])),
                 (255, 0, 0), thickness=2)
    return histImage


ret, frame = cam.read()
while cam.isOpened:
    ret, frame = cam.read()
    if ret:
        start = perf_counter()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # magic happens here
        if flag:
            frame = cv2.equalizeHist(frame)

        histr = cv2.calcHist([frame], [0], None, [256], [0, 256])

        histImage = dark_magic(histr)
        cv2.imshow('Frame', frame)
        cv2.imshow('Hist', histImage)

        end = perf_counter()
        print("OpenCV frame time: " + str(end - start))
        # showing hist
        # plot.set_data(plt.hist(histr))

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

# Closes all the frames
cv2.destroyAllWindows()
