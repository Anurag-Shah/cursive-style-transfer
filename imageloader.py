import numpy as np
import os
import tensorflow as tf
from skimage.transform import resize
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import config
import utils

TARGET_W = config.TARGET_W
TARGET_H = config.TARGET_H
IMG_PER_LANG = config.IMG_PER_LANG
TRAIN_SPLIT = config.TRAIN_SPLIT
input_dir = config.INPUT_DIR
lang_dirs = config.LANG_DIRS
lang_dict = config.LANG_DICT

def getTrainTestData(mode, grayscale=True):
	'''
	Function reads from input data folder
	For each lang in lang_dirs, it:
	- Reads the image
	- Resizes it to target dimensions
	- Adds to train or test data depending on the split
	- Adds label based on mode
		- "type" is standard vs cursive, 0 is standard, 1 is cursive
		- "orth" is orthography name
	'''

	train_data = []
	test_data = []
	train_labels = []
	test_labels = []

	if mode != "type" and mode != "orth":
		raise ValueError("Invalid Mode")

	for lang in lang_dirs:
		dirname = input_dir + "/" + lang
		if not os.path.exists(dirname):
			raise ValueError("Missing Data. Please check the dataset.")
		files = os.listdir(dirname)

		if mode == "type":
			label = np.zeros([2])
			if "Cursive" in lang:
				label[1] = 1
			else:
				label[0] = 1
		else:
			label = np.zeros([8])
			lval = lang_dict.get(lang.split(" ")[0])
			label[lval] = 1
		
		for i, file in enumerate(files):
			img = loadImage(dirname + "/" + file, grayscale)
			if i < TRAIN_SPLIT:
				train_data.append(img)
				train_labels.append(label)
			else:
				test_data.append(img)
				test_labels.append(label)
	train_data = np.array(train_data)
	train_labels = np.array(train_labels)
	test_data = np.array(test_data)
	test_labels = np.array(test_labels)
	return train_data, train_labels, test_data, test_labels

def getDataByType(grayscale=True, onlyLatin=False):
	'''
	Returns train and test data, and labels, as 2 separate lists for each
	This is used for Cyclegan and differes from the previous function which is used for Classifiers/Style Transfer
	'''

	train_data_s = []
	train_data_c = []
	test_data_s = []
	test_data_c = []
	c_flag = True

	for lang in lang_dirs:
		dirname = input_dir + "/" + lang
		if not os.path.exists(dirname):
			raise ValueError("Missing Data. Please check the dataset.")
		files = os.listdir(dirname)

		if "Cursive" in lang:
			c_flag = True
		else:
			c_flag = False
		
		for i, file in enumerate(files):
			img = loadImage(dirname + "/" + file, grayscale, clip=True)
			if i < TRAIN_SPLIT:
				if c_flag:
					train_data_c.append(img)
				else:
					train_data_s.append(img)
			else:
				if c_flag:
					test_data_c.append(img)
				else:
					test_data_s.append(img)

	train_data_s = np.array(train_data_s)
	train_data_c = np.array(train_data_c)
	test_data_s = np.array(test_data_s)
	test_data_c = np.array(test_data_c)

	return train_data_s, train_data_c, test_data_s, test_data_c

def buildTFDataset(grayscale=True, onlyLatin=False):
	'''
	Builds a tf.data.Dataset from the input data
	'''
	train_data_s, train_data_c, test_data_s, test_data_c = getDataByType(grayscale=grayscale, onlyLatin=onlyLatin)
	
	train_d_s = tf.data.Dataset.from_tensor_slices(train_data_s)
	train_d_c = tf.data.Dataset.from_tensor_slices(train_data_c)
	test_d_s = tf.data.Dataset.from_tensor_slices(test_data_s)
	test_d_c = tf.data.Dataset.from_tensor_slices(test_data_c)

	return train_d_s, train_d_c, test_d_s, test_d_c

def loadImage(filename, grayscale, clip=False):
	'''
	Returns an np array given an image
	If clip, clips the image to -1 to 1, and sets the dtype to float
	'''
	if grayscale:
		img = load_img(filename, color_mode="grayscale")
	else:
		img = load_img(filename)

	img = img_to_array(img)

	if clip:
		img = img.astype(np.float32)
		img = utils.clip(img)

	if grayscale:
		img = resize(img, (TARGET_H, TARGET_W, 1))
	else:
		img = resize(img, (TARGET_H, TARGET_W, 3))

	return img