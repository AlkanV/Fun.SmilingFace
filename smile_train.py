#python smile_train.py

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.preprocessing.image import img_to_array
from keras.utils import np_utils
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse
import cv2
import os



from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K



data = []
labels = []

resizedPx = 64


for imagePath in sorted(list(paths.list_images("SMILEs"))):
	image = cv2.imread(imagePath)

	image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	image = cv2.resize(image, (resizedPx, resizedPx)) 

	image = img_to_array(image)
	data.append(image)

	label = imagePath.split(os.path.sep)[-3]
	label = "smiling" if label == "positives" else "not_smiling"
	labels.append(label)





data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

le = LabelEncoder().fit(labels)
labels = np_utils.to_categorical(le.transform(labels), 2)

classTotals = labels.sum(axis=0)
classWeight = classTotals.max() / classTotals



(trainX, testX, trainY, testY) = train_test_split(data,labels, test_size=0.20, stratify=labels, random_state=42)


# create Convolutional Neural Network
model = Sequential()


classes = 2
shapeOfInput = (resizedPx,resizedPx,1)

#convolution layer
model.add(Conv2D(20, (5, 5), padding="same", input_shape=shapeOfInput))
#set the activation function of the convolution layer
model.add(Activation("relu"))
#add pooling layer
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))



model.add(Conv2D(50, (5, 5), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

model.add(Flatten())
model.add(Dense(500))
model.add(Activation("relu"))

# regular multilayer neural network
model.add(Dense(classes))
# the last layer, output layer
model.add(Activation("softmax"))

#since there are two classes, smiling or not smiling, we use binary cross entropy
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])


H = model.fit(trainX, trainY, validation_data=(testX, testY),class_weight=classWeight, batch_size=64, epochs=15, verbose=1)


predictions = model.predict(testX, batch_size=64)
# print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=le.classes_))

# save the model to disk
print("saving the model to disk")
model.save("cnn64.hdf5")

"""

plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, 15), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, 15), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, 15), H.history["acc"], label="acc")
plt.plot(np.arange(0, 15), H.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend()
plt.show()

"""



