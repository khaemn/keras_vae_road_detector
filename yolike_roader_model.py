# https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, Dropout, Reshape, LeakyReLU
from keras.models import Model, load_model
import random
import os
import cv2
import numpy as np
import datetime
from math import sqrt


random.seed(777)

_MODEL_FILENAME = 'models/collab_model_yolike_roader.h5'
_TRAIN_DATA_DIR = 'dataset/train/X'
_WEIGHTS_DIR = 'weights/'

# Input and output images in dataset could be batched to improve IO speed.
# This parameter should be even, and images should be stacked in n*2n pile,
# where 2n is vertical size.
_DATASET_BATCH_SIZE = 1
_HORIZONTAL_BATCH_SIZE = 1 if _DATASET_BATCH_SIZE == 1 else int(sqrt(_DATASET_BATCH_SIZE / 2))
_VERTICAL_BATCH_SIZE = 1 if _DATASET_BATCH_SIZE == 1 else int(_HORIZONTAL_BATCH_SIZE * 2)
if _DATASET_BATCH_SIZE > 1:
    assert _HORIZONTAL_BATCH_SIZE * 2 * _VERTICAL_BATCH_SIZE == _DATASET_BATCH_SIZE

# If true, model is not compiled from scratch> but loaded from the file.
_LOAD_MODEL = True

# When in pretrain run, unlabeled data is used, and therefore mask is ignored.
_PRETRAIN = False

# dimensions of our images.
img_width, img_height = 320, 180  # 160, 90


epochs = 5
iterations = 5
batch_size = 32

# Data loading should be reworked to keras Generators to improve perf.
def load_data(path=_TRAIN_DATA_DIR):
    X, Y = [], []
    logging_limit = 500
    for root_back, dirs_back, files_back in os.walk(path):
        total_files = len(files_back)
        print("Total %u files to load" % total_files)
        total_memory = total_files * img_height * img_width * 2 * 4
        total_mem_mib = int(total_memory / (1024*1024))
        print("Predicted memory consumption %u bytes (%u MBytes)" % (total_memory, total_mem_mib))
        X = np.zeros((total_files, img_height, img_width, 1), dtype='float32')
        adjusted_height = 192  # 96  # !!! to fit with upsampling KERAS layers
        Y = np.zeros((total_files, adjusted_height, img_width, 1), dtype='float32')
        print("Arrays allocated.")
        index = 0
        log_iteration_count = 0
        for _file in files_back:
            image = cv2.imread(os.path.join(path, _file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            for x_index in range(0, _HORIZONTAL_BATCH_SIZE):
                for y_index in range(0, _VERTICAL_BATCH_SIZE):
                    y_offset = img_height * y_index
                    x_offset = img_width * x_index

                    _x = image[y_offset:(y_offset + img_height),
                               x_offset:(x_offset + img_width)]
                    if _PRETRAIN:
                        _y = _x  # Y is the same as X in pretrain run.
                    else:
                        _y = image[y_offset:(y_offset + img_height),
                                  (x_offset + img_width):(x_offset + 2 * img_width)]

                    _x = _x.astype('float32')
                    _x = _x / 255
                    _x = _x.reshape((img_height, img_width, 1))
                    X[index] = _x

                    _y = cv2.resize(_y, (img_width, adjusted_height))
                    _y = _y.astype('float32')
                    _y = _y / 255
                    _y = _y.reshape((adjusted_height, img_width, 1))
                    Y[index] = _y
            index += 1
            log_iteration_count += 1
            if log_iteration_count >= logging_limit:
                log_iteration_count = 0
                print("Loaded %d of %d images %s" % (index, total_files, str(datetime.datetime.now())))
    return X, Y

def get_model():
    # Creating the Model
    # Here, the Model copies the one from tiny YOLO configuration (google "darknet YOLO")
    print("Building model ...")
    model_init = "he_uniform"
    leak_alpha = 0.1

    # input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_1')(input_img)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_2')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_3')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(512, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)

    encoded = Dropout(0.5)(encoded)

    # at this point the representation is (6, 10, 128)

    x = Conv2D(512, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_6')(encoded)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(256, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_4')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_3')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_2')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_1')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer=model_init, padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n" + autoencoder.summary())
    return autoencoder

if __name__ == "__main__":
    if not _LOAD_MODEL:
        autoencoder = get_model()
    else:
        print("Loading model ...")
        autoencoder = load_model(_MODEL_FILENAME)
        print("Model loaded.")

    X, Y = load_data()
    for i in range(0, iterations):
        print("\n\n\n%s\nIteration %d of %d" % (str(datetime.datetime.now()), i+1, epochs))
        autoencoder.fit(X, Y,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1)
        autoencoder.save_weights(_WEIGHTS_DIR + 'roader_yolike_trial_' + str(i) + '.h5')
        autoencoder.save(_MODEL_FILENAME)

    print("\n\n\n%s\nTraining completed." % str(datetime.datetime.now()))

