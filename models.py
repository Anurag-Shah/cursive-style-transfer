import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import config
from layers import ReflectionPadding2D, residual_block, downsample, upsample

imsize = config.IMSIZE
kernel_random_fn = config.KERNEL_RANDOM_FN
res_gamma_fn = config.RES_GAMMA_FN
CHANNELS = config.TARGET_CHANNELS

def get_resnet_generator(filters=64, num_downsampling_blocks=2, num_residual_blocks=9, num_upsample_blocks=2, gamma_initializer=res_gamma_fn, name=None):
	img_input = layers.Input(shape=imsize, name=name + "_img_input")
	x = ReflectionPadding2D(padding=(3, 3))(img_input)
	x = layers.Conv2D(filters, (7, 7), kernel_initializer=kernel_random_fn, use_bias=False)(x)
	x = tfa.layers.InstanceNormalization(gamma_initializer=gamma_initializer)(x)
	x = layers.Activation("relu")(x)

	for i in range(num_downsampling_blocks):
		filters *= 2
		x = downsample(x, filters=filters, activation=layers.Activation("relu"))

	for i in range(num_residual_blocks):
		x = residual_block(x, activation=layers.Activation("relu"))

	for i in range(num_upsample_blocks):
		filters //= 2
		x = upsample(x, filters, activation=layers.Activation("relu"))

	x = ReflectionPadding2D(padding=(3, 3))(x)
	x = layers.Conv2D(CHANNELS, (7, 7), padding="valid")(x)
	x = layers.Activation("tanh")(x)

	model = keras.models.Model(img_input, x, name=name)
	return model

def get_discriminator(filters=64, kernel_initializer=kernel_random_fn, num_downsampling=3, name=None):
	img_input = layers.Input(shape=imsize, name=name + "_img_input")
	x = layers.Conv2D(filters, (4, 4), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(img_input)
	x = layers.LeakyReLU(0.2)(x)

	num_filters = filters
	for num_downsample_block in range(3):
		num_filters *= 2
		if num_downsample_block < 2:
			x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(2, 2))
		else:
			x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(0.2), kernel_size=(4, 4), strides=(1, 1))

	x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(x)

	model = keras.models.Model(inputs=img_input, outputs=x, name=name)
	return model

class CycleGan(keras.Model):
	def __init__(self, generator_G, generator_F, discriminator_X, discriminator_Y, lambda_cycle=8.0, lambda_identity=0.35):
		super(CycleGan, self).__init__()
		self.gen_G = generator_G
		self.gen_F = generator_F
		self.disc_X = discriminator_X
		self.disc_Y = discriminator_Y
		self.lambda_cycle = lambda_cycle
		self.lambda_identity = lambda_identity

	def compile(self, optimizer, gen_loss, disc_loss):
		super(CycleGan, self).compile()
		self.gen_G_optimizer = optimizer
		self.gen_F_optimizer = optimizer
		self.disc_X_optimizer = optimizer
		self.disc_Y_optimizer = optimizer
		self.gen_loss = gen_loss
		self.disc_loss = disc_loss
		self.cycle_loss_fn = keras.losses.MeanAbsoluteError()
		self.identity_loss_fn = keras.losses.MeanAbsoluteError()

	def train_step(self, batch_data):
		real_x, real_y = batch_data

		with tf.GradientTape(persistent=True) as tape:
			fake_y = self.gen_G(real_x, training=True)
			fake_x = self.gen_F(real_y, training=True)

			cycled_x = self.gen_F(fake_y, training=True)
			cycled_y = self.gen_G(fake_x, training=True)

			same_x = self.gen_F(real_x, training=True)
			same_y = self.gen_G(real_y, training=True)

			disc_real_x = self.disc_X(real_x, training=True)
			disc_fake_x = self.disc_X(fake_x, training=True)

			disc_real_y = self.disc_Y(real_y, training=True)
			disc_fake_y = self.disc_Y(fake_y, training=True)

			gen_G_loss = self.gen_loss(disc_fake_y)
			gen_F_loss = self.gen_loss(disc_fake_x)

			cycle_loss_G = self.cycle_loss_fn(real_y, cycled_y) * self.lambda_cycle
			cycle_loss_F = self.cycle_loss_fn(real_x, cycled_x) * self.lambda_cycle

			id_loss_G = self.disc_loss(real_y, same_y) * self.lambda_cycle * self.lambda_identity 
			id_loss_F = self.disc_loss(real_x, same_x) * self.lambda_cycle * self.lambda_identity 

			total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
			total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

			disc_X_loss = self.disc_loss(disc_real_x, disc_fake_x)
			disc_Y_loss = self.disc_loss(disc_real_y, disc_fake_y)

		grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
		grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)
		disc_X_grads = tape.gradient(disc_X_loss, self.disc_X.trainable_variables)
		disc_Y_grads = tape.gradient(disc_Y_loss, self.disc_Y.trainable_variables)

		self.gen_G_optimizer.apply_gradients(zip(grads_G, self.gen_G.trainable_variables))
		self.gen_F_optimizer.apply_gradients(zip(grads_F, self.gen_F.trainable_variables))
		self.disc_X_optimizer.apply_gradients(zip(disc_X_grads, self.disc_X.trainable_variables))
		self.disc_Y_optimizer.apply_gradients(zip(disc_Y_grads, self.disc_Y.trainable_variables))

		return {
			"G_loss": total_loss_G,
			"F_loss": total_loss_F,
			"D_X_loss": disc_X_loss,
			"D_Y_loss": disc_Y_loss,
		}
