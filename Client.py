import sys
sys.path.insert(0, './imagezmq')

import socket
import time
import cv2
from imutils.video import VideoStream
import imagezmq

sender = imagezmq.ImageSender(connect_to='tcp://192.168.1.25:5555')

rpi_name = socket.gethostname()  # send RPi hostname with each image
# picam = VideoStream(usePiCamera=True).start()
cap = cv2.VideoCapture(0)

time.sleep(2.0)  # allow camera sensor to warm up
jpeg_quality = 95  # 0 to 100, higher is better quality, 95 is cv2 default
while True:
    # image = picam.read()
    ret, image = cap.read()
    ret_code, jpg_buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    sender.send_jpg(rpi_name, jpg_buffer)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()