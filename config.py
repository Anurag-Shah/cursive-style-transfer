from tensorflow import keras

'''
Global Settings
'''

# Set this to "type" to classify between standard and cursive, to "orth" to classify between the orthographies
CLASSIFIER_MODE = "type"

# Set this to true for grayscale in classifier and cyclegan. Grayscale has a faster training time and appears to have no negative impact on results
GRAYSCALE_MODE = True

# Set this to true to use only latin instead of every script type. Set it to False for the classifier
CYCLEGAN_ONLY_LATIN = False

# Set this to true to use the 280 word dataset instead of the 30 word for Latin. Set it to False for the classifier
CYCLEGAN_LATIN_EXPANDED = False

'''
Global constants
'''

if CLASSIFIER_MODE == "type":
	OUT_CATEGORIES = 2
elif CLASSIFIER_MODE == "orth":
	OUT_CATEGORIES = 8
else:
	raise ValueError("Invalid mode, please change in config.py")

if GRAYSCALE_MODE:
	TARGET_CHANNELS = 1
else:
	TARGET_CHANNELS = 3

if CYCLEGAN_LATIN_EXPANDED:
	if not CYCLEGAN_ONLY_LATIN:
		raise ValueError("Cannot use the expanded latin dataset without forcing only latin first. Please make the change in config.py")
	CYCLEGAN_OUT_IMAGES = 8
	CYCLEGAN_SHUFFLE_BUFFER = 280
elif CYCLEGAN_ONLY_LATIN:
	CYCLEGAN_OUT_IMAGES = 5
	CYCLEGAN_SHUFFLE_BUFFER = 30
else:
	CYCLEGAN_OUT_IMAGES = 15
	CYCLEGAN_SHUFFLE_BUFFER = 150

if CYCLEGAN_LATIN_EXPANDED:
	INPUT_DIR = "input_latin_expanded"
	IMG_PER_LANG = 280
	TRAIN_SPLIT = 250
	LANG_DIRS = ["Latin Cursive", "Latin Standard"]
else:
	IMG_PER_LANG = 30
	TRAIN_SPLIT = 25
	INPUT_DIR = "input_data"
	LANG_DIRS = ["Arabic Cursive", "Bangla Standard", "Chinese Cursive", "Chinese Standard", "Cyrillic Cursive", "Devanagiri Standard", "Greek Cursive", "Latin Cursive", "Latin Standard", "Telugu Standard"]
'''
Defaults
'''

TARGET_W = 200
TARGET_H = 48
CLASSIFIER_EPOCHS = 30

LANG_DICT = {
	"Arabic": 0,
	"Bangla": 1,
	"Chinese": 2,
	"Cyrillic": 3,
	"Devanagiri": 4,
	"Greek": 5,
	"Latin": 6,
	"Telugu": 7
}

IMSIZE = (TARGET_H, TARGET_W, TARGET_CHANNELS)
KERNEL_RANDOM_FN = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
RES_GAMMA_FN = keras.initializers.RandomNormal(mean=0.0, stddev=0.02)
LOSS_FN = keras.losses.MeanSquaredError()
BATCH_SIZE = 1
CYCLEGAN_EPOCHS = 20
CGLR = 2e-4
BETA = 0.5
DROPRATE = 0.6