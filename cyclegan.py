import numpy as np
import tensorflow as tf
from tensorflow import keras
from imageloader import buildTFDataset
import config
from utils import GanMonitor, unclip, gen_loss, disc_loss
from models import get_resnet_generator, get_discriminator, CycleGan

batch_size = config.BATCH_SIZE
epochs = config.CYCLEGAN_EPOCHS
lr = config.CGLR
beta = config.BETA
grayscale = config.GRAYSCALE_MODE
onlyLatin = config.CYCLEGAN_ONLY_LATIN
out_img = config.CYCLEGAN_OUT_IMAGES
shuffle_buffer_size = config.CYCLEGAN_SHUFFLE_BUFFER

def main():
	print("Loading Data")
	train_std, train_cur, test_std, test_cur = buildTFDataset(grayscale=grayscale, onlyLatin=onlyLatin)
	train_std = train_std.shuffle(shuffle_buffer_size).batch(batch_size)
	train_cur = train_cur.shuffle(shuffle_buffer_size).batch(batch_size)
	test_std = test_std.shuffle(shuffle_buffer_size).batch(batch_size)
	test_cur = test_cur.shuffle(shuffle_buffer_size).batch(batch_size)
	print("Data Loaded")
	print("Beginning Training")

	gen_G = get_resnet_generator(name="generator_G")
	gen_F = get_resnet_generator(name="generator_F")
	disc_X = get_discriminator(name="discriminator_X")
	disc_Y = get_discriminator(name="discriminator_Y")

	cycle_gan_model = CycleGan(generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y)
	opt = keras.optimizers.Adam(learning_rate=lr, beta_1=beta)
	cycle_gan_model.compile(optimizer=opt, gen_loss=gen_loss, disc_loss=disc_loss)
	save_img = GanMonitor(test_std, num_img=out_img)
	cycle_gan_model.fit(tf.data.Dataset.zip((train_std, train_cur)), epochs=epochs, callbacks=[save_img])

	print("Training complete")

if __name__ == "__main__":
	main()