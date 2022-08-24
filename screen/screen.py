from typing import NamedTuple

import cv2
import mss
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI

Rect = NamedTuple("Rect", [("x", int), ("y", int), ("w", int), ("h", int)])

sct = mss.mss()

MAC_SCREEN = (3024, 1964)
MAC_GAME_REGION = Rect(x=495, y=81, w=2522, h=1687)
MAC_SCORING_REGION = Rect(x=0, y=0, w=100, h=100)

DESKTOP_SCREEN = (2560, 1440)
DESKTOP_GAME_REGION = Rect(x=420, y=6, w=2134, h=1428)
DESKTOP_SCORING_REGION = Rect(x=340, y=88, w=48, h=55 * 8)

IN_BLACK = np.array([64], dtype=np.float32)
IN_WHITE = np.array([220], dtype=np.float32)


tesseract_api = PyTessBaseAPI()


class Screen:
    def __init__(self):
        self.identify_screen()

    def identify_screen(self):
        """Returns the screen size and game region"""
        monitor = sct.monitors[1]
        if monitor["width"] == MAC_SCREEN[0]:
            self.game_region = MAC_GAME_REGION
            self.scoring_region = MAC_SCORING_REGION
        else:
            self.game_region = DESKTOP_GAME_REGION
            self.scoring_region = DESKTOP_SCORING_REGION

    def frame(self):
        self.image = np.asarray(sct.grab(sct.monitors[1]))

    def grab(self):
        """Returns a 300x200 image of the game playing area"""
        image = self.image[
            self.game_region.y : self.game_region.y + self.game_region.h,
            self.game_region.x : self.game_region.x + self.game_region.w,
        ]
        image = cv2.resize(image, (300, 200))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Clip for a cleaner image
        image = np.clip((image - IN_BLACK) / (IN_WHITE - IN_BLACK), 0, 255)  # type: ignore
        return image

    def grab_score(self):
        # Grab the scoring region and convert it to grayscale
        image = self.image[
            self.scoring_region.y : self.scoring_region.y + self.scoring_region.h,
            self.scoring_region.x : self.scoring_region.x + self.scoring_region.w,
        ]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("Score", image)
        pil_image = Image.fromarray(image)
        tesseract_api.SetImage(pil_image)
        return tesseract_api.GetUTF8Text()


screen = Screen()
