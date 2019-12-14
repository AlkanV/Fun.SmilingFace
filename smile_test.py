#python smile_test.py -v notsmiling_baby.gif

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2
import os


ap = argparse.ArgumentParser()


ap.add_argument("-v", "--video")
ap.add_argument("-ip", "--ipcam")
args = vars(ap.parse_args())

# load the face detector cascade and smile detector CNN
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

resizedPx = 64
frameWidth = 300


model = load_model("cnn64.hdf5")
ipcam = False



if args.get("camera", False):
	camera = cv2.VideoCapture(0)


elif args.get("video", False):
	camera = cv2.VideoCapture(args["video"])
	frameWidth = 900

elif args.get("ipcam", False):
	print(args["ipcam"])
	camera = cv2.VideoCapture(args["ipcam"])
	ipcam = True
else:
	print("missing video/ipcam argument")
	exit()


while True:
	if ipcam:
		camera.open(args["ipcam"])
		(grabbed, frame) = camera.read()
	else:
		(grabbed, frame) = camera.read()

	if args.get("video") and not grabbed:
		
		extension = os.path.splitext(args.get("video"))[1]
		
		if(extension==".gif"):
			camera = cv2.VideoCapture(args["video"])
			continue

		camera.set(cv2.CAP_PROP_POS_FRAMES,1)
		(grabbed, frame) = camera.read()



	frame = imutils.resize(frame, width=frameWidth)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frameClone = frame.copy()

	#find the face in the video
	rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)

	for (fX, fY, fW, fH) in rects:

		foundFace = gray[fY:fY + fH, fX:fX + fW]
		foundFace = cv2.resize(foundFace, (resizedPx, resizedPx))
		foundFace = foundFace.astype("float") / 255.0
		foundFace = img_to_array(foundFace)
		foundFace = np.expand_dims(foundFace, axis=0)


		#predict
		(fail, success) = model.predict(foundFace)[0]
		# if smiles, back the rectangular green, else red.
		color = (0, 255,0) if success > fail else (0, 0,255)
		cv2.rectangle(frameClone, (fX, fY), (fX + fW, fY + fH),color, 2)


	cv2.imshow("Face", frameClone)
	if cv2.waitKey(1) & 0xFF == ord("q"):
		break

camera.release()
cv2.destroyAllWindows()
