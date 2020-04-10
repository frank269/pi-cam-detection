# pi-cam-detection
Raspberry's Camera Handling project

- Install python
- run: pip install -r requirements.txt
- run on pi: python client.py -s <Server-IP>
- run on server: python server.py

# Download model form link and add it into Models folder
https://drive.google.com/file/d/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-/view

# Align Faces
python FaceNet/align_dataset_mtcnn.py Data/FaceData/raw Data/FaceData/processed --image_size 160 --margin 32 --random_order

# Trainning model
python FaceNet/classifier.py TRAIN Data/FaceData/processed Models/20180402-114759.pb Models/facemodel.pkl


# reset pi-cam
sudo rmmod uvcvideo
sudo modprobe uvcvideo nodrop=1 timeout=5000 quirks=0x80