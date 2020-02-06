import os
import argparse
import numpy as np
import cv2
import time

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True,
	help="name of person need get data")
ap.add_argument("-i", "--interval",
	help="interval to capture image", default=0.1)
args = vars(ap.parse_args())

INTERVAL = args["interval"]
PER_NAME = args["name"]

directory = os.path.join("Data", "FaceData", "raw", PER_NAME)
if not os.path.exists(directory):
    os.makedirs(directory)
cap = cv2.VideoCapture(0)
start_time = time.time()
time.sleep(1)
while True:
    _, image = cap.read()
    if image is not None:
        cv2.imshow("Create Data", image)
        if time.time() - start_time > INTERVAL :
            start_time = time.time()
            filename = os.path.join(directory, "img{}.jpg".format(start_time))
            cv2.imwrite(filename,image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()