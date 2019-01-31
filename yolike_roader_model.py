# https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Activation, Dropout, BatchNormalization, LeakyReLU
from keras.models import Model, load_model
import random
import os
import cv2
import numpy as np
import datetime
from math import sqrt


random.seed(777)

_EXISTING_MODEL_FILENAME = 'models/collab_model_yolike_roader.h5'
_MODEL_FILENAME = 'models/model_yolike_roader.h5'
_TRAIN_DATA_DIR = 'dataset/train/XOUT'
_WEIGHTS_DIR = 'weights/'

# Input and output images in dataset could be batched to improve IO speed.
# This parameter should be even, and images should be stacked in n*2n pile,
# where 2n is vertical size.
_HORIZONTAL_BATCH_SIZE = 10
_VERTICAL_BATCH_SIZE = 20


# If true, model is not compiled from scratch> but loaded from the file.
_LOAD_MODEL = False # True

# When in pretrain run, unlabeled data is used, and therefore mask is ignored.
_PRETRAIN = True

# dimensions of our images.
img_width, img_height = 320, 180  # 160, 90


epochs = 1
iterations = 5
batch_size = 16

# Data loading should be reworked to keras Generators to improve perf.
def load_data(path=_TRAIN_DATA_DIR):
    X, Y = [], []
    _DATA_BATCH_SIZE = _VERTICAL_BATCH_SIZE * _HORIZONTAL_BATCH_SIZE
    logging_limit = int(500 / (_DATA_BATCH_SIZE))
    for root_back, dirs_back, files_back in os.walk(path):
        total_files = len(files_back)
        array_size = _DATA_BATCH_SIZE * total_files
        print("Total %u files to load" % total_files)
        total_memory = array_size * img_height * img_width * 2 * 4
        total_mem_mib = int(total_memory / (1024*1024))
        print("Predicted memory consumption %u bytes (%u MBytes)" % (total_memory, total_mem_mib))
        X = np.zeros((array_size, img_height, img_width, 1), dtype='float32')
        adjusted_height = 180  # 192  # 96  # !!! to fit with upsampling KERAS layers
        Y = np.zeros((array_size, adjusted_height, img_width, 1), dtype='float32')
        print("Arrays allocated, X and Y lengths are %d" % array_size)
        batch_index = 0
        log_iteration_count = 0
        data_pieces_loaded = 0

        debug_index_set = set()
        for _file in files_back:
            image = cv2.imread(os.path.join(path, _file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            for x_index in range(0, _HORIZONTAL_BATCH_SIZE):
                for y_index in range(0, _VERTICAL_BATCH_SIZE):
                    y_offset = img_height * y_index
                    x_offset = img_width * 2 * x_index

                    _x = image[y_offset:(y_offset + img_height),
                               x_offset:(x_offset + img_width)]
                    #if _PRETRAIN:
                    #    _y = _x  # Y is the same as X in pretrain run.
                    #else:
                    _y = image[y_offset:(y_offset + img_height),
                                   (x_offset + img_width):(x_offset + 2 * img_width)]

                    # cv2.imshow("X", _x)
                    # cv2.imshow("Y", _y)
                    # cv2.waitKey(0)

                    index_in_data_array = (batch_index * _DATA_BATCH_SIZE + (y_index * _HORIZONTAL_BATCH_SIZE) + x_index)

                    _x = _x.astype('float32')
                    _x = _x / 255
                    _x = _x.reshape((img_height, img_width, 1))
                    X[index_in_data_array] = _x

                    _y = cv2.resize(_y, (img_width, adjusted_height))
                    _y = _y.astype('float32')
                    _y = _y / 255
                    _y = _y.reshape((adjusted_height, img_width, 1))
                    Y[index_in_data_array] = _y
                    # print("Filled array index %d, batch index %d, x index %d, y index %d"
                    #       % (index_in_data_array, batch_index, x_index, y_index))
                    # debug_index_set.add(index_in_data_array)
                    data_pieces_loaded += 1
            batch_index += 1

            log_iteration_count += 1
            if log_iteration_count >= logging_limit:
                log_iteration_count = 0
                print("Loaded %d of %d images %s" % (batch_index, total_files, str(datetime.datetime.now())))
        print("Loaded %d pieces of data." % data_pieces_loaded)
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

    x = Conv2D(128, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    encoded = Dropout(0.2)(x)

    # at this point the representation is (6, 10, 128)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_6')(encoded)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_5')(x)
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
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder

def get_filter_only_model():
    # Creating the Model
    # Here, the Model copies the one from tiny YOLO configuration (google "darknet YOLO")
    print("Building model ...")
    model_init = "he_uniform"
    leak_alpha = 0.1

    # input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(4, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_1')(input_img)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=leak_alpha)(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(4, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_2')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=leak_alpha)(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(4, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_4')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=leak_alpha)(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_5')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=leak_alpha)(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_6')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    # x = Dropout(0.2)(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_7')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=leak_alpha)(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_8')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=leak_alpha)(x)
    # x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_9')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_10')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=leak_alpha)(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_11')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=leak_alpha)(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Encoder_CONV2D_12')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer=model_init, padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder


def get_asym_model():
    # Creating the Model
    # Here, the Model copies the one from tiny YOLO configuration (google "darknet YOLO")
    print("Building model ...")
    model_init = "he_uniform"
    leak_alpha = 0.1

    # input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format
    # input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(8, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_1_1')(input_img)
    x = Conv2D(8, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_1_2')(input_img)
    x = Conv2D(8, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_1_3')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_2_1')(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_2_2')(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_2_3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), kernel_initializer=model_init, trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = Dropout(0.1)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (6, 10, 128)

    x = Conv2D(256, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_6')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((3, 3))(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((3, 3))(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_4')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((3, 3))(x)

    x = Conv2D(4, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_3')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer=model_init, padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder


if __name__ == "__main__":
    if not _LOAD_MODEL:
        autoencoder = get_asym_model()
    else:
        print("Loading model ...")
        autoencoder = load_model(_EXISTING_MODEL_FILENAME)
        print("Model loaded.")

    X, Y = load_data()
    for i in range(0, iterations):
        print("\n\n\n%s\nIteration %d of %d" % (str(datetime.datetime.now()), i+1, iterations))
        autoencoder.fit(X, Y,
                        epochs=epochs,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1)
        autoencoder.save_weights(_WEIGHTS_DIR + 'roader_yolike_trial_' + str(i) + '.h5')
        autoencoder.save(_MODEL_FILENAME)

    print("\n\n\n%s\nTraining completed." % str(datetime.datetime.now()))

