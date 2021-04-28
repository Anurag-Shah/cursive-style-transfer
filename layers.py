import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_addons as tfa
import config

kernel_random_fn = config.KERNEL_RANDOM_FN
res_gamma_fn = config.RES_GAMMA_FN

class ReflectionPadding2D(layers.Layer):

	def __init__(self, padding=(1, 1), **kwargs):
		self.padding = tuple(padding)
		super(ReflectionPadding2D, self).__init__(**kwargs)

	def call(self, input_tensor, mask=None):
		padding_width, padding_height = self.padding
		padding_tensor = [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]]
		return tf.pad(input_tensor, padding_tensor, mode="REFLECT")

def residual_block(x, activation, kernel_initializer=kernel_random_fn, kernel_size=(3, 3), strides=(1, 1), padding="valid", gamma_initializer=res_gamma_fn):
	dim = x.shape[-1]
	input_tensor = x
	x = ReflectionPadding2D()(input_tensor)
	x = layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding, use_bias=False)(x)
	x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
	x = ReflectionPadding2D()(activation(x))
	x = layers.Conv2D(dim, kernel_size, strides=strides, kernel_initializer=kernel_initializer, padding=padding, use_bias=False)(x)
	x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
	x = layers.add([input_tensor, x])
	return x

def downsample(x, filters, activation, kernel_size=(3, 3), strides=(2, 2)):
	x = layers.Conv2D(filters, kernel_size, strides=strides, kernel_initializer=kernel_random_fn, padding="same", use_bias=False)(x)
	x = tfa.layers.InstanceNormalization(gamma_initializer=res_gamma_fn)(x)
	x = activation(x)
	return x

def upsample(x, filters, activation, kernel_size=(3, 3), strides=(2, 2)):
	x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding="same", kernel_initializer=kernel_random_fn, use_bias=False)(x)
	x = tfa.layers.InstanceNormalization(gamma_initializer=res_gamma_fn)(x)
	x = activation(x)
	return x