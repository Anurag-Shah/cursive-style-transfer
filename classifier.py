import numpy as np
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from imageloader import getTrainTestData
from utils import getAccuracy
import config

mode = config.CLASSIFIER_MODE
TARGET_W = config.TARGET_W
TARGET_H = config.TARGET_H
IMG_PER_LANG = config.IMG_PER_LANG
TRAIN_SPLIT = config.TRAIN_SPLIT
dropRate = config.DROPRATE
grayscale = config.GRAYSCALE_MODE
output_size = config.OUT_CATEGORIES
EPOCHS = config.CLASSIFIER_EPOCHS

def trainconvnet(x, y, epochs):
	categorical_crossentropy = keras.losses.CategoricalCrossentropy()
	lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.001, decay_steps=200, decay_rate = 0.9, staircase=True)
	callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)
	opt = keras.optimizers.Adam(learning_rate = lr_schedule)
	inShape = (TARGET_H, TARGET_W, 1)

	net = keras.models.Sequential()

	net.add(keras.layers.Conv2D(
		32,
		kernel_size=3,
		activation=tf.nn.relu,
		input_shape=inShape,
		padding='same'
		))
	net.add(keras.layers.Conv2D(
		64,
		kernel_size=3,
		activation=tf.nn.relu,
		input_shape=inShape, padding='same'
		))
	net.add(keras.layers.Conv2D(
		64,
		kernel_size=3,
		activation=tf.nn.relu,
		padding='same'
		))
	net.add(keras.layers.MaxPooling2D(pool_size=(3, 3)))
	net.add(tf.keras.layers.Dropout(dropRate))
	net.add(keras.layers.Flatten())
	net.add(keras.layers.BatchNormalization())
	net.add(keras.layers.Dense(512, activation=tf.nn.relu))
	net.add(tf.keras.layers.Dropout(dropRate))
	net.add(keras.layers.Dense(256, activation=tf.nn.relu))
	net.add(tf.keras.layers.Dropout(dropRate))
	net.add(keras.layers.Dense(output_size, activation="softmax"))

	opt = keras.optimizers.Adam(learning_rate = lr_schedule)

	net.compile(optimizer = opt, loss = categorical_crossentropy, metrics = [keras.metrics.CategoricalAccuracy()])
	net.fit(x, y, epochs = epochs, verbose = 1, callbacks = [callback], batch_size=5)

	return net

def main():
	print("Loading Data")
	train_data, train_labels, test_data, test_labels = getTrainTestData(mode=mode, grayscale=grayscale)
	print("Data Loaded")
	print("Training Covnet")
	model = trainconvnet(train_data, train_labels, EPOCHS)
	preds = model.predict(test_data)
	for i in range(preds.shape[0]):
		oneHot = [0] * output_size
		oneHot[np.argmax(preds[i])] = 1
		preds[i] = oneHot
	getAccuracy(test_labels, preds)

if __name__ == "__main__":
	main()