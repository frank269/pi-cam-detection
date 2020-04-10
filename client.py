import sys
sys.path.insert(0, './imagezmq')
import argparse
import socket
import time
import cv2
from imutils.video import VideoStream
import imagezmq


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
	help="ip address of the server to which the client will connect")
args = vars(ap.parse_args())

sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(args["server_ip"]), REQ_REP = True)

rpi_name = socket.gethostname()  # send RPi hostname with each image
# picam = VideoStream(usePiCamera=True).start()
cap = cv2.VideoCapture(0)

time.sleep(2.0)  # allow camera sensor to warm up
jpeg_quality = 95  # 0 to 100, higher is better quality, 95 is cv2 default
while True:
    # image = picam.read()
    ret, image = cap.read()
    if image is None:
        continue
    ret_code, jpg_buffer = cv2.imencode(
        ".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])
    result = sender.send_jpg(rpi_name, jpg_buffer)
    time.sleep(0.3)