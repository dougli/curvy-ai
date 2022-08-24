import time

import cv2

from screen import screen

frames = 0
start_time = time.time()
while True:
    screen.frame()
    image = screen.game_area()
    score = screen.score()
    # cv2.imshow("Curvy", image)
    cv2.waitKey(1)

    now = time.time()
    frames += 1
    fps = round(frames / (now - start_time), 2)
    # print(f"Score: {score} - FPS: {fps}")
