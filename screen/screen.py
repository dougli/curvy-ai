import cv2
import mss
import numpy as np

sct = mss.mss()


def grab():
    return np.asarray(sct.grab(sct.monitors[1]))
