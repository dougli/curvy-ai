import asyncio
import base64
import os
from typing import Any

import cv2
import numpy as np

from .types import GameState

os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1012729"

import secrets

import pyppeteer
from pyppeteer import launch

CURVE_FEVER = "https://curvefever.pro"
# WEB_GL_GAME = "https://playcanv.as/p/LwskqxXT/"


class Game:
    def __init__(self, email: str, password: str, *, headless: bool = True):
        self.email = email
        self.password = password
        self.headless = headless

    async def launch(self):
        args = pyppeteer.defaultArgs({"headless": self.headless})
        args.append("--use-gl=egl")

        self.browser = await launch({"headless": self.headless, "args": args})
        page = await self.browser.newPage()
        await page.goto(CURVE_FEVER)

        # CDP stands for Chrome Devtools Protocol. You can read more about it here:
        # https://chromedevtools.github.io/devtools-protocol/
        self.cdp_session = await page.target.createCDPSession()

        # Use the screencast API to get the game screen.
        self.cdp_session.on("Page.screencastFrame", self._on_screencast_frame)
        self.cdp_session.send(
            "Page.startScreencast",
            {
                "format": "jpeg",
                "quality": 100,
                "maxWidth": 800,
                "maxHeight": 600,
                "everyNthFrame": 1,
            },
        )

        # Click the "SIGN IN" link.
        await page.waitForSelector("a.sign-in")
        await asyncio.sleep(0.5)
        await page.click("a.sign-in")

        # Fill in our username and password and submit.
        await page.waitForSelector("input[name=email]")
        await page.type("input[name=email]", self.email)
        await page.type("input[name=password]", self.password)
        button = await page.xpath("//button[contains(., 'SIGN IN')]")
        await button[0].click()

        # Dismiss an annoying popup that sometimes appears.
        await page.waitForSelector(".popup__x-button", timeout=5000)
        button = await page.querySelector(".popup__x-button")
        if button:
            await button.click()

        # Click the "CREATE MATCH" button.
        await asyncio.sleep(1)
        button = await page.xpath("//button[contains(., 'CREATE MATCH')]")
        await button[0].click()

        # Set up our game parameters and click the "CREATE MATCH" button in the dialog.
        button = await page.xpath("//button[contains(., 'PRIVATE')]")
        await button[0].click()
        await page.type("input[name=password]", secrets.token_urlsafe(8))
        button = await page.xpath("//button[contains(., 'DISABLED')]")
        await button[1].click()  # Disable pickups
        button = await page.xpath(
            "//button[contains(., 'CREATE MATCH')]/div[contains(@class, 'c-button-content')]"
        )
        await button[0].click()  # Click the final "CREATE MATCH" button.

        # Wait for an ad to complete, which is usually 30 seconds.
        await asyncio.sleep(10)
        await page.waitForSelector(".fullscreen-ad-container", hidden=True)

        # Click the "PLAY!" button.
        await page.waitForSelector("span.play-button__content", visible=True)
        await page.click("span.play-button__content")

    @property
    def screen(self):
        return None

    @property
    def state(self) -> GameState:
        return GameState(-1, False, False)

    async def close(self):
        self.cdp_session.send("Page.stopScreencast")
        await self.cdp_session.detach()
        await self.browser.close()

    def _on_screencast_frame(self, params: dict[str, Any]):
        self.cdp_session.send(
            "Page.screencastFrameAck",
            {"sessionId": params["sessionId"]},
        )

        img = base64.b64decode(params["data"])
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)
