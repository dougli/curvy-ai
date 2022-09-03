import asyncio
import base64
import collections
import os
import secrets
import time
from functools import cached_property
from typing import Any, Optional

import cv2
import logger
import numpy as np
from PIL import Image
from termcolor import colored
from tesserocr import PyTessBaseAPI

from .types import Action, Player, Rect

os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1012729"


import pyppeteer
from pyppeteer import launch
from pyppeteer.page import Page

VIEWPORT = {"width": 1280, "height": 720}

# Identifies the play area
PLAY_AREA = Rect(x=200, y=0, w=1080, h=720)
PLAY_AREA_RESIZED = (180, 270)

# Identifies the scoring area
INITIAL_SCORE = 10
SCORE_HEIGHT = 27
SCORE_AREA = Rect(x=170, y=43, w=25, h=SCORE_HEIGHT * 8)
SCORE_ALIVE_COLOR = np.array([60, 46, 39], dtype=np.float32)  # 272e3c
SCORE_DEAD_COLOR = np.array([43, 27, 22], dtype=np.float32)  # 161b2b
SCORE_BACKGROUND_COLOR = np.array([36, 19, 14], dtype=np.float32)  # 0e1324
SCORE_STATE_THRESHOLD = 205

# Color clipping so we get a clean image as opposed to the blue morphy background
IN_BLACK = 60
IN_WHITE = 255

REWARD_ALIVE_PER_SEC = 0.01  # Small bonus every second for staying alive
REWARD_DEAD_PENALTY = 0  # Penalty for dying

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
        self.in_play = False
        self.frame_id = 0
        self.screen_lock = asyncio.Condition()
        self.score = INITIAL_SCORE
        self.last_reward_time = 0

    async def launch(self, url: str) -> None:
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

        logger.info("Loading URL...")
        await self.page.goto(url)

    async def login(self) -> None:
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

    async def create_match(self) -> None:
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

    async def start_match(self) -> None:
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
        self.in_play = True
        asyncio.ensure_future(self._wait_for_game_end())
        logger.success("Game started!")

    @property
    def play_area(self) -> Optional[np.ndarray]:
        screen = self.screen
        if screen is not None:
            # Crop
            image = screen[
                PLAY_AREA.y : PLAY_AREA.y + PLAY_AREA.h,
                PLAY_AREA.x : PLAY_AREA.x + PLAY_AREA.w,
            ]

            # Convert to grayscale
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Resize
            image = (
                image.reshape(PLAY_AREA_RESIZED[0], 4, PLAY_AREA_RESIZED[1], 4)
                .mean(-1, dtype=np.float32)
                .mean(1, dtype=np.float32)
            )

            # Clip away the background
            image = np.clip(
                (image - IN_BLACK) * (255.0 / (IN_WHITE - IN_BLACK)), 0, 255.0
            )

            # Rescale to 0-1 and return as 3D array (for compatibility with the
            # neural network).
            image = image / 255.0
            image = np.expand_dims(image, axis=(0))
            return image
        return None

    @cached_property
    def player(self) -> Player:
        """Returns the current player state. Not thread-safe.

        Player state includes their score and whether we can determine they are alive or
        dead.
        """
        screen = self.screen
        if screen is None:
            return Player(-1, False, False)
        # Grab the scoring region
        image = screen[
            SCORE_AREA.y : SCORE_AREA.y + SCORE_AREA.h,
            SCORE_AREA.x : SCORE_AREA.x + SCORE_AREA.w,
        ]

        # Returns a one-hot vector of the player index if alive, otherwise a zero vector
        is_alive, score = self._ocr_state(image, SCORE_ALIVE_COLOR)
        if is_alive:
            return Player(score, alive=True, dead=False)
        is_dead, score = self._ocr_state(image, SCORE_DEAD_COLOR)
        if is_dead:
            return Player(score, alive=False, dead=True)
        return Player(-1, alive=False, dead=False)

    @property
    def fps(self) -> float:
        if len(self.frame_times) == FPS_COUNTER_SIZE:
            return FPS_COUNTER_SIZE / (self.frame_times[-1] - self.frame_times[0])
        return 0

    async def set_action(self, action: Action) -> None:
        if action == self.current_action:
            return

        key_events = []
        if self.current_action != Action.NOTHING:
            key_events.append(self.page.keyboard.up(self.current_action.key))
        if action != Action.NOTHING:
            key_events.append(self.page.keyboard.down(action.key))

        await asyncio.gather(*key_events)
        self.current_action = action

    async def step(self, action: Action) -> tuple[float, bool]:
        """Take a step in the environment.

        Args:
            action (Action): The action to take.

        Returns:
            tuple[float, bool]: The reward and whether the game is over.
        """
        if not self.in_play:
            return 0, True

        curr_state = self.player

        if curr_state.dead:
            return 0, True

        if not curr_state.alive:
            # If we are neither dead nor alive, we are in one of the following states:
            #
            # 1. The game is in play and we are alive, but the score ranking has changed
            #    and for a few frames we cannot accurately determine the score or state.
            #    This will only happen in multiplayer games.
            #
            # 2. The run has ended because everyone else has died. This will only happen
            #    in multiplayer games.
            #
            # 3. The game has crashed, disconnected, or for whatever reason something
            #    awful happened, and we are no longer in a game.
            #
            # 4. The game canvas has rendered the game but the game hasn't started yet.
            #
            # States 1 and 2 are only possible in multiplayer games. State 3 should be
            # handled by _wait_for_game_end(), and state 4 should be handled by the
            # caller, awaiting for wait_for_alive() before calling step().
            pass

        # Perform the input
        await self.set_action(action)

        # Wait for a new frame
        async with self.screen_lock:
            await self.screen_lock.wait()

        next_state = self.player

        # Calculate the reward
        now = time.time()
        elapsed = now - self.last_reward_time
        self.last_reward_time = now

        score_reward = 0
        alive_reward = (
            next_state.alive * REWARD_ALIVE_PER_SEC * elapsed
            + next_state.dead * REWARD_DEAD_PENALTY
        )

        if next_state.score != -1:
            score_reward = next_state.score - self.score
            self.score = next_state.score
        final_reward = score_reward + alive_reward

        return final_reward, False

    async def wait_for_alive(self):
        """Wait for the player to be alive."""
        while True:
            player = self.player
            if player.alive:
                self.last_reward_time = time.time()
                return

            # Wait for a new frame
            async with self.screen_lock:
                await self.screen_lock.wait()

    async def close(self):
        self.cdp.send("Page.stopScreencast")
        await self.cdp.detach()
        await self.browser.close()

    def _on_screencast_frame(self, params: dict[str, Any]) -> None:
        self.frame_id += 1
        self.frame_times.append(time.time())
        self.cdp.send(
            "Page.screencastFrameAck",
            {"sessionId": params["sessionId"]},
        )

        image = base64.b64decode(params["data"])
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        asyncio.ensure_future(self._update_screen(image))

    async def _update_screen(self, image) -> None:
        async with self.screen_lock:
            self.screen = image
            self.screen_lock.notify_all()
        if hasattr(self, "player"):
            del self.player

        if self.show_screen:
            cv2.imshow("Screen", image)
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
        wait_for_canvas = self.page.waitForSelector("canvas", hidden=True, timeout=0)
        wait_for_rematch = self.page.waitForSelector(".rematch-container", timeout=0)

        done, pending = await asyncio.wait(
            [wait_for_canvas, wait_for_rematch], return_when=asyncio.FIRST_COMPLETED
        )
        self.in_play = False
        self.score = INITIAL_SCORE
        for task in pending:
            task.cancel()
        await self.set_action(Action.NOTHING)

        if wait_for_rematch in done:
            logger.success("Game ended smoothly! Starting a new game...")
            await asyncio.sleep(0.5)
            await self.page.click(".rematch-container")
            await self.start_match()
            return

        logger.warning("Game ended suddenly! Canvas window was killed. Refreshing...")
        await self.page.reload()
        await self.create_match()
        await self.start_match()
