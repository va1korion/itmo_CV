import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numba import njit

plot = plt.subplot(1, 1, 1)
plt.ion()


def just_show(frame) -> (np.array, np.array):
    hist, bins = np.histogram(frame.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    # plt.plot(cdf_normalized, color='b')
    # plt.hist(frame.flatten(), 256, [0, 256], color='r')
    # plt.xlim([0, 256])
    # plt.legend(('cdf', 'histogram'), loc='upper left')

    return frame, frame.flatten()


@njit()
def equalize(frame) -> (np.array, np.array):
    hist, bins = np.histogram(frame.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    frame = cdf[frame]
    return frame, frame.flatten()


cam = cv2.VideoCapture("./examples/video_example.mp4")
flag = False


# working the first frame
ret, frame = cam.read()
if ret:
    histr = cv2.calcHist([frame], [0], None, [256], [0, 256])
    cv2.imshow('Frame', frame)
    # showing hist
    # plot = plot.imshow()


while cam.isOpened:
    ret, frame = cam.read()
    if ret:

        frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # equalizing if flag is set
        if flag:
            frame, hist = equalize(frame)

        image = Image.fromarray(frame.astype(np.uint8))
        cv2.imshow('Frame', frame)

        # showing hist
        # plot.set_data(plt.hist(histr))

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('h'):
            flag = not flag
        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break


    else:
        break

cam.release()
plt.ioff()
cv2.destroyAllWindows()
