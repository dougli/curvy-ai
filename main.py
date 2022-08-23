import time

import cv2

import screen

start = time.time()
while True:
    image = screen.grab()
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Curvy", image)
    cv2.waitKey(1)
    print("FPS: {}".format(1 / (time.time() - start)))
    start = time.time()
