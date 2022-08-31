import asyncio
import base64
import io
import os
import time
from typing import Any

import cv2
import numpy as np
from PIL import Image

os.environ["PYPPETEER_CHROMIUM_REVISION"] = "1012729"

import pyppeteer
from pyppeteer import launch


async def main():
    args = pyppeteer.defaultArgs({"headless": True})
    args.append("--use-gl=egl")

    browser = await launch({"headless": True, "args": args})
    page = await browser.newPage()
    # Some random WebGL game so we can test frame rate
    await page.goto("https://playcanv.as/p/LwskqxXT/")

    cdp_session = await page.target.createCDPSession()

    global start_time
    start_time = time.time()

    def onFrameHandler(params: dict[str, Any], *args, **kwargs):
        global start_time

        cdp_session.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})

        img = base64.b64decode(params["data"])
        img = np.frombuffer(img, dtype=np.uint8)
        img = cv2.imdecode(img, cv2.IMREAD_COLOR)

        cv2.imshow("img", img)
        cv2.waitKey(1)

        end = time.time()
        fps = 1 / (end - start_time)
        print(f"FPS: {fps}, sessionId: {params['sessionId']}")
        start_time = end

    cdp_session.on("Page.screencastFrame", onFrameHandler)
    cdp_session.send(
        "Page.startScreencast",
        {
            "format": "jpeg",
            "quality": 100,
            "maxWidth": 800,
            "maxHeight": 600,
            "everyNthFrame": 1,
        },
    )

    await asyncio.sleep(15)
    await browser.close()


asyncio.get_event_loop().run_until_complete(main())
