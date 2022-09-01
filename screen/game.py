import asyncio
import base64
import collections
import os
import secrets
import time
from typing import Any, Optional

import cv2
import logger
import numpy as np
from PIL import Image
from termcolor import colored
from tesserocr import PyTessBaseAPI

from .types import Action, GameState, Rect

os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1012729"


import pyppeteer
from pyppeteer import launch
from pyppeteer.page import Page

CURVE_FEVER = "https://curvefever.pro"
WEB_GL_GAME = "https://playcanv.as/p/LwskqxXT/"

VIEWPORT = {"width": 1280, "height": 720}

PLAY_AREA = Rect(x=200, y=0, w=1080, h=720)
PLAY_AREA_RESIZED = (144, 216)

SCORE_HEIGHT = 27
SCORE_AREA = Rect(x=170, y=43, w=25, h=SCORE_HEIGHT * 8)
SCORE_ALIVE_COLOR = np.array([60, 46, 39], dtype=np.float32)  # 272e3c
SCORE_DEAD_COLOR = np.array([43, 27, 22], dtype=np.float32)  # 161b2b
SCORE_BACKGROUND_COLOR = np.array([36, 19, 14], dtype=np.float32)  # 0e1324
SCORE_STATE_THRESHOLD = 205

FPS_COUNTER_SIZE = 10


tesseract_api = PyTessBaseAPI()


class Game:
    def __init__(
        self,
        email: str,
        password: str,
        *,
        headless: bool = True,
        show_screen: bool = False
    ):
        self.email = email
        self.password = password
        self.headless = headless
        self.show_screen = show_screen

        self.screen = None
        self.frame_times = collections.deque(maxlen=FPS_COUNTER_SIZE)
        self.current_action = Action.NOTHING

    async def launch(self):
        args = pyppeteer.defaultArgs({"headless": self.headless})
        args.append("--use-gl=egl")

        self.browser = await launch({"headless": self.headless, "args": args})
        self.page = await self.browser.newPage()
        await self.page.setViewport(VIEWPORT)

        # Use the Chrome Devtools Protocol screencast API to get the game screen.
        # You can read more about it here:
        # https://chromedevtools.github.io/devtools-protocol/
        self.cdp = await self.page.target.createCDPSession()
        self.cdp.on("Page.screencastFrame", self._on_screencast_frame)
        self.cdp.send(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": 100,
                "maxWidth": VIEWPORT["width"],
                "maxHeight": VIEWPORT["height"],
                "everyNthFrame": 1,
            },
        )

        # await self.page.goto(WEB_GL_GAME)
        logger.info("Loading Curve Fever Pro...")
        await self.page.goto(CURVE_FEVER)

    async def start_game(self) -> None:
        """Start a game. Clicks through all of the menus to get the game to start."""

        # Click the "SIGN IN" link.
        logger.info("Signing in...")
        await self.page.waitForSelector("a.sign-in")
        await asyncio.sleep(0.5)
        await self.page.click("a.sign-in")

        # Fill in our username and password and submit.
        await self.page.waitForSelector("input[name=email]")
        await self.page.type("input[name=email]", self.email)
        await self.page.type("input[name=password]", self.password)
        button = await self.page.xpath("//button[contains(., 'SIGN IN')]")
        await button[0].click()

        # Dismiss an annoying popup that sometimes shows up.
        logger.info("Dismissing annoying popup...")
        await self.page.waitForSelector(".popup__x-button", timeout=5000)
        button = await self.page.querySelector(".popup__x-button")
        if button:
            await button.click()

        # Click the "CREATE MATCH" button.
        logger.info("Creating a match...")
        await asyncio.sleep(1)
        button = await self.page.xpath("//button[contains(., 'CREATE MATCH')]")
        await button[0].click()

        # Set up our game parameters and click the "CREATE MATCH" button in the dialog.
        button = await self.page.xpath("//button[contains(., 'PRIVATE')]")
        await button[0].click()
        await self.page.type("input[name=password]", secrets.token_urlsafe(8))
        button = await self.page.xpath("//button[contains(., 'DISABLED')]")
        await button[1].click()  # Disable pickups
        button = await self.page.xpath(
            "//button[contains(., 'CREATE MATCH')]/div[contains(@class, 'c-button-content')]"
        )
        await button[0].click()  # Click the final "CREATE MATCH" button.

        # Wait for an ad to complete, which is usually 30 seconds.
        logger.info("Waiting for ad to complete (takes over 30s)...")
        await asyncio.sleep(10)
        await self.page.waitForSelector(".fullscreen-ad-container", hidden=True)

        logger.info("Checking to make sure we are not a spectator...")
        try:
            button = await self.page.xpath(
                "//button[contains(., 'Play in this match')]",
                timeout=1000,
            )
            if button:
                await button[0].click()
        except:
            # Button was not found. Don't worry, just move on.
            pass

        # Click the "PLAY!" button.
        logger.info("Skipping powerup selection...")
        await self.page.waitForSelector("span.play-button__content", visible=True)
        await self.page.click("span.play-button__content")

        logger.info("Clicking the play button...")
        await self.page.waitForSelector("button.button--start-timer")
        await asyncio.sleep(0.5)
        await self.page.click("button.button--start-timer")

        await self.page.waitForSelector("canvas")
        asyncio.ensure_future(self._wait_for_game_end())
        logger.success("Game started!")

    @property
    def play_area(self) -> Optional[np.ndarray]:
        if self.screen is not None:
            image = self.screen[
                PLAY_AREA.y : PLAY_AREA.y + PLAY_AREA.h,
                PLAY_AREA.x : PLAY_AREA.x + PLAY_AREA.w,
            ]

            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            image = (
                image.reshape(PLAY_AREA_RESIZED[0], 5, PLAY_AREA_RESIZED[1], 5)
                .mean(-1, dtype=np.float32)
                .mean(1, dtype=np.float32)
                .astype(np.uint8)
            )

            return image
        return None

    @property
    def state(self) -> GameState:
        if self.screen is None:
            return GameState(-1, False, False)
        # Grab the scoring region
        image = self.screen[
            SCORE_AREA.y : SCORE_AREA.y + SCORE_AREA.h,
            SCORE_AREA.x : SCORE_AREA.x + SCORE_AREA.w,
        ]

        # Returns a one-hot vector of the player index if alive, otherwise a zero vector
        is_alive, score = self._ocr_state(image, SCORE_ALIVE_COLOR)
        if is_alive:
            return GameState(score, alive=True, dead=False)
        is_dead, score = self._ocr_state(image, SCORE_DEAD_COLOR)
        if is_dead:
            return GameState(score, alive=False, dead=True)
        return GameState(-1, alive=False, dead=False)

    @property
    def fps(self) -> float:
        if len(self.frame_times) == FPS_COUNTER_SIZE:
            return FPS_COUNTER_SIZE / (self.frame_times[-1] - self.frame_times[0])
        return 0

    async def set_action(self, action: Action) -> None:
        if action == self.current_action:
            return
        if self.current_action != Action.NOTHING:
            await self.page.keyboard.up(self.current_action.key)
        if action != Action.NOTHING:
            await self.page.keyboard.down(action.key)
        self.current_action = action

    async def close(self):
        self.cdp.send("Page.stopScreencast")
        await self.cdp.detach()
        await self.browser.close()

    def _on_screencast_frame(self, params: dict[str, Any]) -> None:
        self.cdp.send(
            "Page.screencastFrameAck",
            {"sessionId": params["sessionId"]},
        )

        image = base64.b64decode(params["data"])
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        self.screen = image

        self.frame_times.append(time.time())

        if self.show_screen:
            cv2.imshow("Screen", self.screen)
            cv2.waitKey(1)

    def _ocr_state(self, image, color) -> tuple[bool, int]:
        """Returns a tuple, where the first element is True if the player is in that
        state, and the second element is the player's score"""
        vertical_strip = image[:, -1:, :3]
        mask = cv2.inRange(vertical_strip, color - 4, color + 4)
        mask = cv2.resize(mask, (1, 8), interpolation=cv2.INTER_AREA)
        cv2.threshold(mask, SCORE_STATE_THRESHOLD, 1, cv2.THRESH_BINARY, mask)
        is_state = np.max(mask)

        if is_state:
            player_index = np.argmax(mask)
            # Crop the image to the player's score
            score_image = image[
                player_index * SCORE_HEIGHT : (player_index + 1) * SCORE_HEIGHT, :
            ]
            pil_image = Image.fromarray(score_image)
            tesseract_api.SetImage(pil_image)
            try:
                return True, int(tesseract_api.GetUTF8Text())
            except ValueError:
                return True, -1
        return False, -1

    async def _wait_for_game_end(self):
        # Wait indefinitely for the game to end.
        await self.page.waitForSelector("canvas", hidden=True, timeout=0)
        logger.warning("Game ended!")
        self.screen = None
