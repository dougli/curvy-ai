import cv2
import mss
import numpy as np

sct = mss.mss()

MAC_SCREEN = (3024, 1964)
MAC_GAME_REGION = {"left": 495, "top": 81, "width": 2522, "height": 1687}

DESKTOP_SCREEN = (2560, 1440)
DESKTOP_GAME_REGION = {"left": 420, "top": 6, "width": 2134, "height": 1428}


def grab():
    """Returns a 300x200 image of the game playing area"""
    monitor = sct.monitors[1]
    if monitor["width"] == MAC_SCREEN[0]:
        image = sct.grab(MAC_GAME_REGION)
    else:
        image = sct.grab(DESKTOP_GAME_REGION)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.resize(image, (300, 200))
