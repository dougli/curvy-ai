import asyncio
import base64
import collections
import os
import time
from functools import cached_property
from typing import Any, Optional

import cv2
import logger
import numpy as np
from PIL import Image
from tesserocr import PyTessBaseAPI

from .types import Action, Player, Rect

os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1012729"


import pyppeteer
import pyppeteer.errors
from pyppeteer import launch
from pyppeteer.page import Page

# =================================================== 6x
VIEWPORT = {"width": 1920, "height": 1080}
PLAY_AREA = Rect(x=816, y=0, w=600, h=1080)
PLAY_AREA_SCALING_FACTOR = 10
SCORE_HEIGHT = 41
SCORE_AREA = Rect(x=253, y=66, w=41, h=SCORE_HEIGHT * 8)

# =================================================== 5x
# VIEWPORT = {"width": 1600, "height": 900}
# PLAY_AREA_SCALING_FACTOR = 5
# PLAY_AREA = Rect(x=250, y=0, w=1350, h=900)
# SCORE_HEIGHT = 34
# SCORE_AREA = Rect(x=211, y=55, w=34, h=SCORE_HEIGHT * 8)

# =================================================== 4x
# VIEWPORT = {"width": 1280, "height": 720}
# PLAY_AREA = Rect(x=200, y=0, w=1080, h=720)
# PLAY_AREA_SCALING_FACTOR = 4
# SCORE_HEIGHT = 27
# SCORE_AREA = Rect(x=170, y=43, w=25, h=SCORE_HEIGHT * 8)

PLAY_AREA_RESIZED = (108, 60)
assert PLAY_AREA.w / PLAY_AREA_SCALING_FACTOR == PLAY_AREA_RESIZED[1]
assert PLAY_AREA.h / PLAY_AREA_SCALING_FACTOR == PLAY_AREA_RESIZED[0]

# Identifies the scoring area
SCORE_ALIVE_COLOR = np.array([60, 46, 39], dtype=np.float32)  # 272e3c
SCORE_DEAD_COLOR = np.array([43, 27, 22], dtype=np.float32)  # 161b2b
SCORE_BACKGROUND_COLOR = np.array([36, 19, 14], dtype=np.float32)  # 0e1324
SCORE_STATE_THRESHOLD = 205

# Color clipping so we get a clean image as opposed to the blue morphy background
IN_BLACK = 48
IN_WHITE = 90

REWARD_ALIVE_PER_SEC = 0  # Small bonus every second for staying alive
REWARD_DEAD_PENALTY = -10  # Penalty for dying

INITIAL_SCORE = 10
FPS_COUNTER_SIZE = 10

MAX_ALIVE_WAIT_TIME = 15  # Seconds to wait for the game to start


tesseract_api = PyTessBaseAPI()


class Game:
    def __init__(
        self,
        match_name: str,
        match_password: str,
        *,
        headless: bool = True,
        show_screen: bool = False,
        show_play_area: bool = False,
    ):
        self.match_name = match_name
        self.match_password = match_password

        self.headless = headless
        self.show_screen = show_screen
        self.show_play_area = show_play_area

        self.frame_times = collections.deque(maxlen=FPS_COUNTER_SIZE)
        self.current_action = Action.NOTHING
        self.in_play = False
        self.screen_lock = asyncio.Condition()
        self.score = INITIAL_SCORE
        self.last_reward_time = 0
        self.last_alive = 0

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
                "quality": 90,
                "maxWidth": VIEWPORT["width"],
                "maxHeight": VIEWPORT["height"],
                "everyNthFrame": 5,
            },
        )

        logger.info("Loading URL...")
        await self.page.goto(url)

    async def login(self, email: str, password: str) -> None:
        # Click the "SIGN IN" link.
        logger.info("Signing in...")
        await self.page.waitForSelector("a.sign-in")
        await asyncio.sleep(1.5)
        await self.page.click("a.sign-in")

        # Fill in our username and password and submit.
        await self.page.waitForSelector("input[name=email]")
        await self.page.type("input[name=email]", email)
        await self.page.type("input[name=password]", password)
        button = await self.page.xpath("//button[contains(., 'SIGN IN')]")
        await button[0].click()

        # Dismiss an annoying popup that sometimes shows up.
        logger.info("Dismissing annoying popup...")
        try:
            await self.page.waitForSelector(".popup__x-button", timeout=5000)
            button = await self.page.querySelector(".popup__x-button")
            if button:
                await button.click()
        except pyppeteer.errors.TimeoutError:
            # Sometimes this popup doesn't show up.
            pass
        await asyncio.sleep(1)

    async def create_match(self) -> None:
        # Click the "CREATE MATCH" button.
        logger.info(f"Creating match '{self.match_name}'...")
        button = await self.page.xpath("//button[contains(., 'CREATE MATCH')]")
        await button[0].click()

        # Set up our game parameters and click the "CREATE MATCH" button in the dialog.
        button = await self.page.xpath("//button[contains(., 'UNRANKED')]")
        await button[0].click()
        button = await self.page.xpath("//button[contains(., 'PRIVATE')]")
        await button[0].click()
        await self.page.type("input[name=password]", self.match_password)
        button = await self.page.xpath("//button[contains(., 'DISABLED')]")
        await button[1].click()  # Disable pickups
        button = await self.page.xpath(
            "//button[contains(., 'CREATE MATCH')]/div[contains(@class, 'c-button-content')]"
        )
        await button[0].click()  # Click the final "CREATE MATCH" button.

    async def join_match(self) -> None:
        # Click the "JOIN MATCH" button.
        logger.info(f"Joining match '{self.match_name}'...")
        button = await self.page.xpath("//button[contains(., 'JOIN MATCH')]")
        await button[0].click()

        # Join the match with the given name and password
        start = time.time()
        while time.time() - start < 30:
            button = await self.page.xpath(
                f'//*[contains(text(), "{self.match_name}")]/..//button'
            )
            if button:
                await button[0].click()
                break

            await self.page.waitForSelector("span.refresh-icon")
            button = await self.page.querySelector("span.refresh-icon")
            await button.click()
            await asyncio.sleep(3)

        await asyncio.sleep(3)
        await self.page.type("input[name=password]", self.match_password)
        button = await self.page.xpath("//button[contains(., 'JOIN MATCH')]")
        await button[0].click()  # Click the final "JOIN MATCH" button.

    async def skip_powerup(self) -> None:
        # Wait for an ad to complete, which is usually 30 seconds.
        logger.info("Waiting for ad to complete (takes over 30s)...")
        await asyncio.sleep(5)
        await self.page.waitForSelector(
            ".fullscreen-ad-container", hidden=True, timeout=40000
        )

        logger.info("Checking to make sure we are not a spectator...")
        try:
            await self.page.waitForXPath(
                "//button[contains(., 'Play in this match')]", timeout=2000
            )
            button = await self.page.xpath(
                "//button[contains(., 'Play in this match')]"
            )
            if button:
                await button[0].click()
        except:
            # Button was not found. Don't worry, just move on.
            pass

        logger.info("Skipping powerup selection...")
        await self.page.waitForSelector("span.play-button__content", visible=True)
        await self.page.click("span.play-button__content")

    async def wait_for_player_ready(self, username: str) -> None:
        logger.info(f"Waiting for player '{username}' to join...")
        await self.page.xpath(f'//*[contains(text(), "{username}")]')

    async def start_match(self) -> None:
        logger.info("Clicking the play button...")
        await self.page.waitForSelector("button.button--start-timer:not([disabled])")
        await asyncio.sleep(0.5)
        await self.page.click("button.button--start-timer")

    async def wait_for_start(self) -> None:
        await self.page.waitForSelector("canvas", timeout=60000)
        self.in_play = True
        self.score = INITIAL_SCORE
        asyncio.ensure_future(self._wait_for_game_end())
        logger.info("Game started!")

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
                image.reshape(
                    PLAY_AREA_RESIZED[0],
                    PLAY_AREA_SCALING_FACTOR,
                    PLAY_AREA_RESIZED[1],
                    PLAY_AREA_SCALING_FACTOR,
                )
                .mean(-1, dtype=np.float32)
                .mean(1, dtype=np.float32)
            )

            # Clip away the background
            image = np.clip(
                (image - IN_BLACK) * (255.0 / (IN_WHITE - IN_BLACK)), 0, 255.0
            )

            if self.show_play_area:
                cv2.imshow("image", image.astype(np.uint8))
                cv2.waitKey(1)

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
            logger.error("Calling 'step' when game is not in play!")
            return 0, True

        # Perform the input
        await self.set_action(action)

        # Wait for a new frame
        async with self.screen_lock:
            await self.screen_lock.wait()

        next_state = self.player

        if next_state.alive:
            self.last_alive = time.time()
        elif not next_state.dead and self.last_alive < time.time() - 1:
            logger.error("Player is not alive nor dead for 1 second. Ending round.")
            return 0, True

        # Calculate the reward
        now = time.time()
        elapsed = now - self.last_reward_time
        self.last_reward_time = now

        score_reward = 0
        won = False
        if next_state.score != -1:
            score_reward = next_state.score - self.score
            self.score = next_state.score
        if score_reward == 10:
            won = True
            # Wait 1 extra second for the UI to set the winning player to not alive
            await asyncio.sleep(1)

        alive_reward = (
            next_state.alive * REWARD_ALIVE_PER_SEC * elapsed
            + next_state.dead * REWARD_DEAD_PENALTY * (1 - won)
        )
        final_reward = score_reward + alive_reward

        return final_reward, next_state.dead or won

    async def wait_for_alive(self) -> bool:
        """Wait for the player to be alive."""
        while True:
            if not self.in_play:
                return False
            player = self.player
            if player.alive:
                self.last_alive = time.time()
                self.last_reward_time = time.time()
                return True

            # Wait for a new frame
            async with self.screen_lock:
                await self.screen_lock.wait()

    async def close(self):
        self.cdp.send("Page.stopScreencast")
        await self.cdp.detach()
        await self.browser.close()

    @cached_property
    def screen(self):
        image = base64.b64decode(self._frame_data)
        image = np.frombuffer(image, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image

    def _on_screencast_frame(self, params: dict[str, Any]) -> None:
        self.frame_times.append(time.time())
        self.cdp.send(
            "Page.screencastFrameAck",
            {"sessionId": params["sessionId"]},
        )

        asyncio.ensure_future(self._update_frame(params["data"]))

    async def _update_frame(self, frame_data) -> None:
        async with self.screen_lock:
            self._frame_data = frame_data
            if hasattr(self, "screen"):
                del self.screen
            if hasattr(self, "player"):
                del self.player
            self.screen_lock.notify_all()

        if self.show_screen:
            cv2.imshow("Screen", self.screen)
            cv2.waitKey(1)

    def _ocr_state(self, image, color) -> tuple[bool, int]:
        """Returns a tuple, where the first element is True if the player is in that
        state, and the second element is the player's score"""
        vertical_strip = image[:, -1:, :3]
        mask = cv2.inRange(vertical_strip, color - 6, color + 6)
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
        for task in pending:
            task.cancel()
        await self.set_action(Action.NOTHING)

        if wait_for_rematch in done:
            logger.warning("Game ended smoothly! Starting a new game...")
        else:
            logger.warning("Game ended! Starting a new game...")

        await asyncio.sleep(1)
        await self.page.click(".rematch-container")
