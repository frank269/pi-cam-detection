import cv2
import numpy as np

class Detection:
    def __init__(self, prototxtPath = "./faceHandling/models/deploy.prototxt.txt",  modelPath = "./faceHandling/models/res10_300x300_ssd_iter_140000.caffemodel", confident = 0.95):
        print("Face Detection Init!")
        self.net = cv2.dnn.readNetFromCaffe(prototxt=prototxtPath, caffeModel=modelPath)
        self.confident = confident

    def getFace(self,frame):
        (h, w) = frame.shape[:2]
        # find faces from frame
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        boxes = []
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence < self.confident:
                continue
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
		    # (startX, startY, endX, endY) = box.astype("int")
            boxes.append(box.astype("int"))
        # faces = frame
        # boxes = [0,0,0,0]
        return boxes
