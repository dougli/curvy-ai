import asyncio
import time
from typing import Any

import pyppeteer
from pyppeteer import launch

start_time = time.time()
frames = 0


async def main():
    args = pyppeteer.defaultArgs({"headless": True})
    args.append("--use-gl=egl")
    print(args)
    browser = await launch({"headless": True, "args": args})
    page = await browser.newPage()
    await page.goto(
        # "https://media3.giphy.com/media/RR1hAXyHKS0K4oGQ0z/giphy.gif?cid=ecf05e47c60461f6869e8ba8d1d8f5ea9e1a7bac635fc7cb&rid=giphy.gif&ct=g"
        "https://playcanv.as/p/LwskqxXT/"
        # "chrome://gpu"
    )

    target = page.target
    cdp_session = await target.createCDPSession()

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
    global start_time
    start_time = time.time()

    def onFrameHandler(params: dict[str, Any], *args, **kwargs):
        global frames, start_time
        keys = [*params.keys()]
        cdp_session.send("Page.screencastFrameAck", {"sessionId": params["sessionId"]})
        end = time.time()
        fps = 1 / (end - start_time)
        bytes = len(params["data"])
        print(f"FPS: {fps}, sessionId: {params['sessionId']}, bytes: {bytes}")
        start_time = end

    cdp_session.on("Page.screencastFrame", onFrameHandler)

    await asyncio.sleep(15)
    await browser.close()


asyncio.get_event_loop().run_until_complete(main())
