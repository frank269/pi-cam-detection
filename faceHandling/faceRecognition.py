from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import argparse
from FaceNet import facenet
import imutils
import os
import sys
import math
import pickle
from FaceNet.align import detect_face
import numpy as np
import cv2
import collections
from sklearn.svm import SVC
from .config import *

class Recognition:

    def __init__(self, sess):
        self.sess = sess
        print("Face Recognition Init!")
        with open(CLASSIFIER_PATH, 'rb') as file:
            self.model, self.class_names = pickle.load(file)
        print("Custom Classifier, Successfully loaded")
        
        # Load the model
        print('Loading feature extraction model')
        facenet.load_model(FACENET_MODEL_PATH)

        # Get input and output tensors
        self.images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        self.embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        self.phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        self.embedding_size = self.embeddings.get_shape()[1]
        self.pnet, self.rnet, self.onet = detect_face.create_mtcnn(self.sess, "FaceNet/align")
        print("Feature extraction, Successfully loaded")

    def recognize_face(self, frame):
        # recognize id of faces
        frame = imutils.resize(frame, width=600)
        frame = cv2.flip(frame, 1)
        bounding_boxes, _ = detect_face.detect_face(frame, MINSIZE, self.pnet, self.rnet, self.onet, THRESHOLD, FACTOR)
        faces_found = bounding_boxes.shape[0]
        try:
            if faces_found > 0:
                det = bounding_boxes[:, 0:4]
                bb = np.zeros((faces_found, 4), dtype=np.int32)
                for i in range(faces_found):
                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]
                    if (bb[i][3]-bb[i][1])/frame.shape[0]>0.25:
                        cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                        scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE),
                                            interpolation=cv2.INTER_CUBIC)
                        scaled = facenet.prewhiten(scaled)
                        scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                        feed_dict = {self.images_placeholder: scaled_reshape, self.phase_train_placeholder: False}
                        emb_array = self.sess.run(self.embeddings, feed_dict=feed_dict)

                        predictions = self.model.predict_proba(emb_array)
                        best_class_indices = np.argmax(predictions, axis=1)
                        best_class_probabilities = predictions[
                            np.arange(len(best_class_indices)), best_class_indices]
                        best_name = self.class_names[best_class_indices[0]]
                        # print("Name: {}, Probability: {}".format(best_name, best_class_probabilities))
                        if best_class_probabilities > MATCH_PROBABILITIES:
                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 0, 255), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            name = self.class_names[best_class_indices[0]]
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y + 17),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)
                        else:
                            name = "Unknown"
                        return name, (bb[i][0], bb[i][1], bb[i][2], bb[i][3]), frame

        except:
            pass
        return None, None, frame

    def recognize_emotion(self, face):
        
        # recognize emotion faces
        emotion = "Happy"
        return emotion
