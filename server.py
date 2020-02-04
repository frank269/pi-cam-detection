import sys
sys.path.insert(0, './imagezmq')

import argparse
import numpy as np
import cv2
import imagezmq
import time
from faceHandling.faceDetection import *
from faceHandling.faceRecognition import *

# ap = argparse.ArgumentParser()
# ap.add_argument("-p", "--prototxt", required=True,
# 	help="path to Caffe 'deploy' prototxt file")
# ap.add_argument("-m", "--model", required=True,
# 	help="path to Caffe pre-trained model")
# ap.add_argument("-c", "--confidence", type=float, default=0.2,
# 	help="minimum probability to filter weak detections")
# ap.add_argument("-mW", "--montageW", required=True, type=int,
# 	help="montage frame width")
# ap.add_argument("-mH", "--montageH", required=True, type=int,
# 	help="montage frame height")
# args = vars(ap.parse_args())

image_hub = imagezmq.ImageHub()
detection = Detection()
recognition = Recognition()

start_time = time.time()
while True:
    rpi_name, jpg_buffer = image_hub.recv_jpg()
    image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
    image_hub.send_reply(b'OK')
    if image is not None:
        boxes = detection.getFace(image)
        name = recognition.recognize_face(image)
        emotion = recognition.recognize_emotion(image)

        for (startX, startY, endX, endY) in boxes:
            text = name
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(image, (startX, startY), (endX, endY),
                (0, 0, 255), 2)
            cv2.putText(image, text, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

        cv2.putText(image, rpi_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow(rpi_name, image)
        # if time.time() - start_time > 1 :
        #     start_time = time.time()
        #     filename = "./faceHandling/data1/img{}.jpg".format(start_time)
        #     cv2.imwrite(filename,image)
    cv2.waitKey(1)