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


UPLOAD_FOLDER = 'Data/FaceData/raw'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])
MINIMUM_NUMBER_IMAGE = 15
app = Flask(__name__)
app.json_encoder = MyJsonEncoder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024


def align_face():
    print("start align")
    align_face = AlignFace()
    align_face.start_align()

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
        return jsonify(Server_Response(False,'Text not found!',None, start_time))
    

    return jsonify(Server_Response(True,'Play success!',None, start_time))

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='API Run configuration')
    parser.add_argument('--port', type=int, default=5000)
    parser.add_argument('--debug', type=bool, default= False)

    args = parser.parse_args()
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)