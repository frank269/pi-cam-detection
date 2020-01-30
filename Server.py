import sys
sys.path.insert(0, './imagezmq')

import numpy as np
import cv2
import imagezmq
import time

image_hub = imagezmq.ImageHub()
while True:
    rpi_name, jpg_buffer = image_hub.recv_jpg()
    image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
    if image is not None:
        cv2.imshow(rpi_name, image)
    cv2.waitKey(1)
    image_hub.send_reply(b'OK')