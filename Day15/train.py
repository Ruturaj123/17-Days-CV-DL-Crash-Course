import matplotlib
matplotlib.use("Agg")
 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from pokemon.smallerVGGNet import SmallerVGGNet
import matplotlib.pyplot as plt
from imutils import paths
import numpy as np
import argparse
import random
import pickle
import cv2
import os

epochs = 100
learning_rate = 1e-3
batch_size = 32
image_dims = (96, 96, 3)

train = []
test = []

image_paths = sorted(list(paths.list_images('dataset')))
random.seed(19)
random.shuffle(image_paths)

for image_path in image_paths:
	image = cv2.imread(image_path)
	image = cv2.resize(image, (image_dims[1], image_dims[0]))
	image = img_to_array(image)
	train.append(image)

	label = image_path.split(os.path.sep)[-2]
	test.append(label)

train = np.array(train, dtype='float') / 255.0
test = np.array(test)

labelBinarizer = LabelBinarizer()
test = labelBinarizer.fit_transform(test)

(X_train, X_test, y_train, y_test) = train_test_split(train, test, test_size=0.2, random_state=19)

datagen = ImageDataGenerator(rotation_range=25, width_shift_range=0.1, height_shift_range=0.1,
							 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode="nearest")


model = SmallerVGGNet.build(width = image_dims[1], height = image_dims[0], depth = image_dims[2],
						 	classes = len(labelBinarizer.classes_))
optimizer = Adam(lr = learning_rate, decay = learning_rate / epochs)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])

M = model.fit_generator(datagen.flow(X_train, y_train, batch_size = batch_size), validation_data = (X_test, y_test),
					steps_per_epoch = len(X_train), epochs = epochs)

model.save('pokedex.model')

f = open('labelBinarizer.pickle', 'wb')
f.write(pickle.dumps(labelBinarizer))
f.close()

plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, epochs), M.history["loss"], label="train_loss")
plt.plot(np.arange(0, epochs), M.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, epochs), M.history["acc"], label="train_acc")
plt.plot(np.arange(0, epochs), M.history["val_acc"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="upper left")
plt.savefig(args["plot"])