# pi-cam-detection
Raspberry's Camera Handling project

- Install python
- run: pip install -r requirements.txt
- run on pi: python client.py -s <Server-IP>
- run on server: python server.py

# Download model form link and add it into Models folder
https://doc-04-5s-docs.googleusercontent.com/docs/securesc/v5cn886kngi09jl64c1btiultego0ojr/46gnnjtgin16ec1p0niu11b9d10vtorj/1580378400000/18056234690049221457/00605056220972093873/1EXPBSXwTaqrSC0OhUdXNmKSh9qJUQ55-?e=download&authuser=0&nonce=3tak4j1go3hg4&user=00605056220972093873&hash=086t707426p1qi6sn8fesreem2e7stlc

# Align Faces
python FaceNet\align_dataset_mtcnn.py Data\FaceData\raw Data\FaceData\processed --image_size 160 --margin 32 --random_order

# Trainning model
python FaceNet\classifier.py TRAIN Data\FaceData\processed Models\20180402-114759.pb Models\facemodel.pkl