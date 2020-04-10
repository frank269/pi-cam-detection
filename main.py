import os
from flask import Flask, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import argparse
import time
from flask.json import JSONEncoder
from enum import Enum
import json
from FaceNet.face_align import AlignFace
from FaceNet.facenet import *
from FaceNet.face_classifier import *
from flask_cors import CORS
import requests
import base64

import sys
sys.path.insert(0, './imagezmq')
import numpy as np
import cv2
import imagezmq
from faceHandling.faceRecognition import *
# from outputHandling.textToSpeak import TTS
from inputHandling.userInfomation import User
from datetime import datetime
import threading
from scipy import misc

class Server_Response(object):
    def __init__(self, success, msg, data, start_time):
        self.success = success
        self.msg = msg
        self.data = data
        self.excution_time = time.time() - start_time

class MyJsonEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Server_Response):
            return {
                'Success' : obj.success,
                'Msg' : obj.msg,
                'Data' : obj.data,
                'Execution_time' : obj.excution_time
            }
        return super(MyJsonEncoder, self).default(obj)

class Validate_Face_Status(Enum):
    VALID = 1
    INVALID = 2

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class StoppableThread(threading.Thread):
    """Thread class with a stop() method. The thread itself has to check
    regularly for the stopped() condition."""

    def __init__(self,  *args, **kwargs):
        super(StoppableThread, self).__init__(*args, **kwargs)
        self._stop_event = threading.Event()

    def stop(self):
        self._stop_event.set()

    def stopped(self):
        return self._stop_event.is_set()

image_hub = imagezmq.ImageHub()
SERVER_URL = "http://192.168.1.25:4000/userActivity/create"
CLIENT_URL = "http://192.168.1.12:1234/"
# tts = TTS()

def save_checkin_image(id, image):
    directory = os.path.join("Data", "Checkin", id)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = os.path.join(directory, datetime.now().strftime("%m_%d_%Y_%H_%M_%S") + ".jpg")
    print(filename)
    cv2.imwrite(filename, image)

def checkin(id, image):
    # tts.play_ding()
    url = SERVER_URL
    bytes_data = image.reshape(-1)
    base64_str = str(base64.b64encode(bytes_data), 'utf-8')
    body = json.dumps({'userId': id, 'realTime' : datetime.now().strftime("%H:%M:%S"), 'image' : base64_str})
    print(id)
    headers = {'content-type': 'application/json'}
    r = requests.post(url, data=body, headers=headers)
    res = r.json()
    # print(r.text)
    # print(res)
    if res['data'] != '':
        # print(res['data'])
        requests.get(CLIENT_URL + "play_text", {'text':res['data']})
    else:
        requests.get(CLIENT_URL + "play")
    return r.text

recognition = Recognition()
def run():
    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            recognition.loadModel()
            recognition.setup_Enviroment(sess)
            start_time = time.time()
            last_name = -1
            match_count = 0
            while True:
                if recognition.isEnviromentOk:
                    try:
                        rpi_name, jpg_buffer = image_hub.recv_jpg()
                        image_hub.send_reply(b'OK')
                        image = cv2.imdecode(np.frombuffer(jpg_buffer, dtype='uint8'), -1)
                        if image is not None:
                            name, bbox, image_recog = recognition.recognize_face(image)
                            # emotion = recognition.recognize_emotion(image)

                            if name is not None:
                                # print(name, bbox)
                                if name == last_name and last_name != 'Unknown':
                                    match_count += 1
                                    if match_count > 3 and time.time() - start_time > 8:
                                        checkin(name, image[bbox[1]:bbox[3], bbox[0]:bbox[2], :])
                                        save_checkin_image(name, image)
                                        match_count = 0
                                        start_time = time.time()

                                else:
                                    match_count = 0
                                    last_name = name
                                    # save_checkin_image(name, image)
                        cv2.imshow(rpi_name, image_recog)
                        cv2.waitKey(1)
                    except:
                        print("Error")
                    # cv2.putText(image, rpi_name, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    # cv2.imshow(rpi_name, image)
                # cv2.waitKey(1)
                else: 
                    print("reload")
                    recognition.loadModel()
    # print("goodbye!")


UPLOAD_FOLDER = 'Data/FaceData/raw'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
MINIMUM_NUMBER_IMAGE = 15
app = Flask(__name__)
CORS(app)
app.json_encoder = MyJsonEncoder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route("/align", methods= ['GET'])
def align_face():
    print("start align")
    AlignFace().start_align()
    return "done"

@app.route("/train", methods= ['GET'])
def train_face():
    print("start train")
    TrainingFace().train()
    return "done"

def validate_face(user_id):
    dirName = os.path.join(app.config['UPLOAD_FOLDER'],user_id)
    if not os.path.exists(dirName):
        return Validate_Face_Status.INVALID
    num_faces = len([f for f in os.listdir(dirName)if os.path.isfile(os.path.join(dirName, f))])
    if num_faces < MINIMUM_NUMBER_IMAGE:
        return Validate_Face_Status.INVALID
    return Validate_Face_Status.VALID

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/play_tts", methods= ['POST'])
def play_TTS():
    start_time = time.time()
    data = json.loads(request.data)
    text = data.get("text",None)
    if text is None:
        return ""
        # tts.speak(text)
    return jsonify(Server_Response(True,'Play success!',None, start_time))

t = StoppableThread(target = run) 
def restart_recog_thread():
    global recognition
    recognition.isEnviromentOk = False

@app.route("/register_user", methods= ['GET', 'POST'])
def register_user():
    if request.method == 'POST':
        start_time = time.time()
        user_id = request.form.get('id')
        if user_id is None or user_id == '':
            return jsonify(Server_Response(False,'Missing User\'s ID!',None, start_time))
        if 'images' not in request.files:
            return jsonify( Server_Response(False, 'File Not found!', None, start_time))
        user_images = request.files.getlist('images')
        for image in user_images:
            if image.filename is None or image.filename == '':
                return jsonify(Server_Response(False,'File Not found!',None, start_time))
            filename = secure_filename(image.filename)
            dirName = os.path.join(app.config['UPLOAD_FOLDER'],user_id)
            if not os.path.exists(dirName):
                os.mkdir(dirName)
            filepath = os.path.join(dirName,filename)
            image.save(filepath)
        print('upload complete!')
        print('align face started!')
        align_face()
        status = validate_face(user_id)
        if status == Validate_Face_Status.INVALID:
            return jsonify(Server_Response(True,'Not enough face data for this id!',None, start_time))
        return jsonify(Server_Response(True,'Register User Successful!',None, start_time))
    return '''
    <form method="POST" enctype="multipart/form-data" action="/register_user">
        <label for="fname">User's ID: </label>
        <input type="text" name="id" id="id">
        <br/>
        <label for="fname">User's Image: </label>
        <input type="file" name="images" multiple accept='image/*'>
        <br/>
        <input type="submit" value="register">
    </form>
    '''

@app.route("/reload", methods= ['GET'])
def reload_recognize_server():
    restart_recog_thread()
    return "ok"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='API Run configuration')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', type=bool, default= False)
    args = parser.parse_args()
    t.start()
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)