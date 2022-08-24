import time

import cv2

from screen import screen

frames = 0
start_time = time.time()
while True:
    screen.frame()
    image = screen.game_area()
    score = screen.game_state()

    now = time.time()
    frames += 1
    fps = round(frames / (now - start_time), 2)
    print(f"FPS: {fps}")
