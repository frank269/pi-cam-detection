import cv2

class Detection:
    def __init__(self):
        print("Face Detection Init!")

    def getFace(self,frame):
        
        # find faces from frame
        faces = frame
        boxes = [0,0,0,0]

        return faces, boxes
