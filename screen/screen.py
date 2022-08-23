import mss
import numpy as np

sct = mss.mss()

DESKTOP_SCREEN_SIZE = (2560, 1440)
GAME_LEFT_OFFSET = 414
GAME_BORDER = 6

GAME_REGION = {
    "left": GAME_LEFT_OFFSET + GAME_BORDER,
    "top": GAME_BORDER,
    "width": DESKTOP_SCREEN_SIZE[0] - GAME_LEFT_OFFSET - GAME_BORDER * 2,
    "height": DESKTOP_SCREEN_SIZE[1] - GAME_BORDER * 2,
}


def grab():
    return np.asarray(sct.grab(GAME_REGION))
