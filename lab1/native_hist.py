import cv2
import numpy as np
from time import perf_counter

def just_show(frame) -> (np.array, np.array):
    hist, bins = np.histogram(frame.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    return frame, hist


def equalize(frame) -> (np.array, np.array):
    hist, bins = np.histogram(frame.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_m = np.ma.masked_equal(cdf, 0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m, 0).astype('uint8')
    frame = cdf[frame]
    return frame, hist


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


cam = cv2.VideoCapture("./examples/video_example.mp4")
flag = False


# working the first frame
ret, frame = cam.read()

while cam.isOpened:
    ret, frame = cam.read()
    if ret:
        start = perf_counter()
        frame = np.array(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        # magic happens here
        if flag:
            frame, hist = equalize(frame)
        else:
            frame, hist = just_show(frame)

        histImage = dark_magic(hist)
        cv2.imshow('Frame', frame)
        cv2.imshow('Hist', histImage)

        end = perf_counter()
        print("Native frame time: "+str(end - start))

        if cv2.waitKey(25) & 0xFF == ord('h'):
            flag = not flag
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        # Break the loop
    else:
        break

cam.release()
