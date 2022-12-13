import cv2
import numpy as np
import matplotlib.pyplot as plt
from numba import njit
from time import perf_counter

plot = plt.subplot(1, 1, 1)
plt.ion()


# one stupid way to draw hists
def dark_magic(histr):
    histr = histr.reshape(256, 1)
    histr = 255 * histr / histr[-1]
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


def just_show(frame) -> (np.array, np.array):
    hist, bins = np.histogram(frame.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    return frame, hist


def equalize(frame):
    cdf, hist = njhist(frame)

    cdf_m = np.ma.masked_equal(cdf, 0)  # masked equal not supported by numba for some reason

    cdf_m = eq(cdf_m)

    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    result = cdf[frame]
    return result, hist


@njit()
def njhist(frame):
    hist, bins = np.histogram(frame.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    return cdf, hist


@njit()
def eq(cdf_m):
    cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min())

    return cdf_m


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
        start = perf_counter()
        frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # equalizing if flag is set
        if flag:
            frame, hist = equalize(frame)
        else:
            frame, hist = just_show(frame)

        histImage = dark_magic(hist)
        cv2.imshow('Frame', frame)
        cv2.imshow('Hist', histImage)
        # plot.set_data(plt.hist(histr))
        end = perf_counter()
        print("Numba frame time: " + str(end - start))
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
