import time

import cv2

import screen

start = time.time()
while True:
    image = screen.grab()
    cv2.imshow("Curvy", image)
    cv2.waitKey(1)

    print("FPS: {}".format(1 / (time.time() - start)))
    start = time.time()
