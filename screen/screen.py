import cv2
import mss
import numpy as np

sct = mss.mss()

MAC_SCREEN = (3024, 1964)
MAC_GAME_REGION = {"left": 495, "top": 81, "width": 2522, "height": 1687}

DESKTOP_SCREEN = (2560, 1440)
DESKTOP_GAME_REGION = {"left": 420, "top": 6, "width": 2134, "height": 1428}

IN_BLACK = np.array([64], dtype=np.float32)
IN_WHITE = np.array([220], dtype=np.float32)


def identify_screen():
    """Returns the screen size and game region"""
    monitor = sct.monitors[1]
    if monitor["width"] == MAC_SCREEN[0]:
        return MAC_GAME_REGION
    return DESKTOP_GAME_REGION


game_screen = identify_screen()


def grab():
    """Returns a 300x200 image of the game playing area"""
    image = sct.grab(game_screen)
    image = np.asarray(image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (300, 200))

    # Clip for a cleaner image
    image = np.clip((image - IN_BLACK) / (IN_WHITE - IN_BLACK), 0, 255)  # type: ignore
    return image
