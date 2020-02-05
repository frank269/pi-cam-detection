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

with tf.Graph().as_default():
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
    with sess.as_default():
        image_hub = imagezmq.ImageHub()
        # detection = Detection()
        recognition = Recognition(sess)

        start_time = time.time()
        while True:
            rpi_name, jpg_buffer = image_hub.recv_jpg()
            image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
            image_hub.send_reply(b'OK')
            if image is not None:
                name, bbox, image = recognition.recognize_face(image)
                # emotion = recognition.recognize_emotion(image)

                if name is not None:
                    print(name, bbox)

                # cv2.putText(image, rpi_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow(rpi_name, image)
            cv2.waitKey(1)