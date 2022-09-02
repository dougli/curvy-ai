import cv2
import mss
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI

from .types import Player, Rect

sct = mss.mss()

MAC_SCREEN = (3024, 1964)
MAC_GAME_REGION = Rect(x=495, y=81, w=2522, h=1687)
MAC_SCORING_REGION = Rect(x=0, y=0, w=100, h=100)

DESKTOP_SCREEN = (2560, 1440)
# DESKTOP_GAME_REGION = Rect(x=420, y=6, w=2134, h=1428)
DESKTOP_GAME_REGION = Rect(x=400, y=0, w=2160, h=1440)
DESKTOP_SCORING_REGION = Rect(x=340, y=88, w=48, h=55 * 8)
GAME_AREA_SIZE = (180, 270)


IN_BLACK = np.array([64], dtype=np.float32)
IN_WHITE = np.array([220], dtype=np.float32)

SCORE_ALIVE_COLOR = np.array([60, 46, 39], dtype=np.float32)  # 272e3c
SCORE_DEAD_COLOR = np.array([43, 27, 22], dtype=np.float32)  # 161b2b
SCORE_BACKGROUND_COLOR = np.array([36, 19, 14], dtype=np.float32)  # 0e1324
SCORE_STATE_THRESHOLD = 205


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

    def game_area(self):
        """Returns a 300x200 image of the game playing area"""
        image = self.image[
            self.game_region.y : self.game_region.y + self.game_region.h,
            self.game_region.x : self.game_region.x + self.game_region.w,
        ]

        if self.game_region == DESKTOP_GAME_REGION:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = (
                image.reshape(GAME_AREA_SIZE[0], 8, GAME_AREA_SIZE[1], 8)
                .mean(-1, dtype=np.float32)
                .mean(1, dtype=np.float32)
                .astype(np.uint8)
            )
        else:
            image = cv2.resize(image, GAME_AREA_SIZE, interpolation=cv2.INTER_AREA)

        # Clip for a cleaner image
        image = np.clip((image - IN_BLACK) / (IN_WHITE - IN_BLACK), 0, 255)  # type: ignore
        return image

    def game_state(self) -> Player:
        # Grab the scoring region
        image = self.image[
            self.scoring_region.y : self.scoring_region.y + self.scoring_region.h,
            self.scoring_region.x : self.scoring_region.x + self.scoring_region.w,
        ]

        # Returns a one-hot vector of the player index if alive, otherwise a zero vector
        is_alive, score = self._check_state(image, SCORE_ALIVE_COLOR)
        if is_alive:
            return Player(score, alive=True, dead=False)
        is_dead, score = self._check_state(image, SCORE_DEAD_COLOR)
        if is_dead:
            return Player(score, alive=False, dead=True)
        return Player(-1, alive=False, dead=False)

    def _check_state(self, image, color) -> tuple[bool, int]:
        """Returns a tuple, where the first element is True if the player is in that
        state, and the second element is the player's score"""
        vs = image[:, -1:, :3]
        mask = cv2.inRange(vs, color - 4, color + 4)
        mask = cv2.resize(mask, (1, 8), interpolation=cv2.INTER_AREA)
        cv2.threshold(mask, SCORE_STATE_THRESHOLD, 1, cv2.THRESH_BINARY, mask)
        is_state = np.max(mask)

        if is_state:
            player_index = np.argmax(mask)
            # Crop the image to the player's score
            score_image = image[player_index * 55 : (player_index + 1) * 55, :]
            pil_image = Image.fromarray(score_image)
            tesseract_api.SetImage(pil_image)
            try:
                return True, int(tesseract_api.GetUTF8Text())
            except ValueError:
                return True, -1
        return False, -1


screen = Screen()
