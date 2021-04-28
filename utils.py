import numpy as np
import tensorflow as tf
from tensorflow import keras
import config

def getAccuracy(labels, preds):
	acc = 0
	for i in range(preds.shape[0]):
		if np.array_equal(preds[i], labels[i]):
			acc = acc + 1
	accuracy = acc / preds.shape[0]
	print("Classifier accuracy: %f%%" % (accuracy * 100))

class GanMonitor(keras.callbacks.Callback):
	def __init__(self, test_std, num_img=4):
		self.num_img = num_img
		self.test_std = test_std

	def on_epoch_end(self, epoch, logs=None):
		for i, img in enumerate(self.test_std.take(self.num_img)):
			prediction = self.model.gen_G(img)[0].numpy()
			prediction = (unclip(prediction)).astype(np.uint8)
			prediction = keras.preprocessing.image.array_to_img(prediction)
			prediction.save("generated_images/generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch + 1))

def clip(x):
	return (x / 127.5) - 1.0

def unclip(x):
	return (x * 127.5) + 127.5

def gen_loss(fake):
	fake_loss = config.LOSS_FN(tf.ones_like(fake), fake)
	return fake_loss

def disc_loss(real, fake):
	real_loss = config.LOSS_FN(tf.ones_like(real), real)
	fake_loss = config.LOSS_FN(tf.zeros_like(fake), fake)
	return (real_loss + fake_loss) * 0.5