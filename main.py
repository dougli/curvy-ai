import time

import cv2

from screen import screen

last_frame = 0
frames = 0
start_time = time.time()
while True:
    screen.frame()
    image = screen.grab()
    score = screen.grab_score()
    cv2.imshow("Curvy", image)
    cv2.waitKey(1)

    now = time.time()
    frames += 1
    fps = round(frames / (now - start_time), 2)
    print(f"Score: {score} - FPS: {fps}")
    # last_frame = time.time()
