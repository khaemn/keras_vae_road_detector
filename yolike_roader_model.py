# https://blog.keras.io/building-autoencoders-in-keras.html

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D,\
                         Activation, Dropout, BatchNormalization, LeakyReLU,\
                         Conv2DTranspose, Flatten, Reshape
from keras.models import Model, load_model, Sequential
import random
import os
import cv2
import numpy as np
import datetime
import gc
import statistics
from time import sleep

random.seed(777)

_EXISTING_MODEL_FILENAME = 'models/micro2_model_yolike_roader.h5'
_MODEL_FILENAME = 'models/micro2_model_yolike_roader.h5'
_TRAIN_DATA_DIR = '/home/rattus/Projects/PythonNN/datasets/3-TRAIN'
_PRETRAIN_DATA_DIR = 'D:/__PROJECTS/PythonNN/keras_vae_road_detector/data/test1'
_PRETRAIN_DATA_DIR2 = 'D:/__PROJECTS/PythonNN/keras_vae_road_detector/data/test2'
_PRETRAIN_DATA_DIR3 = 'D:/__PROJECTS/PythonNN/keras_vae_road_detector/data/test3'
_WEIGHTS_DIR = 'weights/'
_WEIGHTS_FILE = _WEIGHTS_DIR + 'exp_vae_roader_pretrain.h5'

# Input and output images in dataset could be batched to improve IO speed.
# This parameter should be even, and images should be stacked in n*2n pile,
# where 2n is vertical size.
_HORIZONTAL_BATCH_SIZE = 50  # 15
_VERTICAL_BATCH_SIZE = 60  # 40

_PRETRAIN_HORIZONTAL_BATCH_SIZE = 40
_PRETRAIN_VERTICAL_BATCH_SIZE = 40


# If true, model is not compiled from scratch> but loaded from the file.
_LOAD_MODEL = True

# When in pretrain run, unlabeled data is used, and therefore mask is ignored.
_PRETRAIN = False

# dimensions of our images.
img_width, img_height = 320, 180  # 160, 90


_epochs = 3
_iterations = 3
_batch_size = 32

# Data loading should be reworked to keras Generators to improve perf.
def load_data(folder=None, single_file=None):
    X, Y = [], []
    _DATA_BATCH_SIZE = _VERTICAL_BATCH_SIZE * _HORIZONTAL_BATCH_SIZE
    logging_limit = int(500 / (_DATA_BATCH_SIZE))

    textures = []
    if single_file is None and folder is not None:
        for root_back, dirs_back, files_back in os.walk(folder):
            for _file in files_back:
                textures.append(_file)
    else:
        textures = [single_file]
        folder = ''

    total_files = len(textures)
    array_size = _DATA_BATCH_SIZE * total_files
    total_memory = array_size * img_height * img_width * 12 # 12 is exp. coeff.
    total_mem_mib = int(total_memory / (1024*1024))  #
    print("Total %u files to load, predicted memory consumption %u bytes (%u MBytes)"
          % (total_files, total_memory, total_mem_mib))
    X = np.zeros((array_size, img_height, img_width, 1), dtype='float32')
    adjusted_height = 192  # 192  # 96  # !!! to fit with upsampling KERAS layers
    adjusted_width = 320
    Y = np.zeros((array_size, adjusted_height, adjusted_width, 1), dtype='float32')
    # print("Arrays allocated, X and Y lengths are %d" % array_size)
    batch_index = 0
    log_iteration_count = 0
    data_pieces_loaded = 0

    debug_index_set = set()
    for _file in textures:
        image = cv2.imread(os.path.join(folder, _file))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        for x_index in range(0, _HORIZONTAL_BATCH_SIZE):
            for y_index in range(0, _VERTICAL_BATCH_SIZE):
                y_offset = img_height * y_index
                x_offset = img_width * 2 * x_index

                _x = image[y_offset:(y_offset + img_height),
                           x_offset:(x_offset + img_width)]

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

                _y = cv2.resize(_y, (adjusted_width, adjusted_height))
                _y = _y.astype('float32')
                _y = _y / 255
                _y = _y.reshape((adjusted_height, adjusted_width, 1))
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

def load_pretrain_data(path=_PRETRAIN_DATA_DIR):
    _DATA_BATCH_SIZE = _PRETRAIN_VERTICAL_BATCH_SIZE * _PRETRAIN_HORIZONTAL_BATCH_SIZE
    logging_limit = int(500 / (_DATA_BATCH_SIZE))
    for root_back, dirs_back, files_back in os.walk(path):
        total_files = len(files_back)
        array_size = _DATA_BATCH_SIZE * total_files
        print("Total %u files to load" % total_files)
        total_memory = array_size * img_height * img_width * 2 * 4
        total_mem_mib = int(total_memory / (1024 * 1024))
        print("Predicted memory consumption %u bytes (%u MBytes)" % (total_memory, total_mem_mib))

        X = np.zeros((array_size, img_height, img_width, 1), dtype='float32')

        adjusted_height = 192  # 192  # 96  # !!! to fit with upsampling KERAS layers
        Y = np.zeros((array_size, adjusted_height, img_width, 1), dtype='float32')

        print("Array allocated, X length is %d" % array_size)
        batch_index = 0
        log_iteration_count = 0
        data_pieces_loaded = 0

        debug_index_set = set()
        for _file in files_back:
            image = cv2.imread(os.path.join(path, _file))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            for x_index in range(0, _PRETRAIN_HORIZONTAL_BATCH_SIZE):
                for y_index in range(0, _PRETRAIN_VERTICAL_BATCH_SIZE):
                    y_offset = img_height * y_index
                    x_offset = img_width * x_index

                    _x = image[y_offset:(y_offset + img_height),
                         x_offset:(x_offset + img_width)]
                    _y = _x.copy()

                    # cv2.imshow("X", _x)
                    # cv2.waitKey(0)

                    index_in_data_array =\
                        (batch_index * _DATA_BATCH_SIZE + (y_index * _PRETRAIN_HORIZONTAL_BATCH_SIZE) + x_index)

                    _x = _x.astype('float32')
                    _x = _x / 255
                    _x = _x.reshape((img_height, img_width, 1))
                    X[index_in_data_array] = _x

                    _y = cv2.resize(_y, (img_width, adjusted_height))
                    _y = _y.astype('float32')
                    _y = _y / 255
                    _y = _y.reshape((adjusted_height, img_width, 1))
                    Y[index_in_data_array] = _y

                    data_pieces_loaded += 1
            batch_index += 1

            log_iteration_count += 1
            if log_iteration_count >= logging_limit:
                log_iteration_count = 0
                print("Loaded %d of %d images %s" % (batch_index, total_files, str(datetime.datetime.now())))
        print("Loaded %d pieces of data." % data_pieces_loaded)
    return X, Y


def get_vae_model():
    print("Building model ...")
    model_init = "he_uniform"
    leak_alpha = 0.1
    dropout = 0.2

    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format

    x = BatchNormalization()(input_img)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init,
               padding='same', name='Encoder_CONV2D_1')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init,
               padding='same', name='Encoder_CONV2D_2')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init,
               padding='same', name='Encoder_CONV2D_3')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = Dropout(dropout)(x)

    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init,
               padding='same', name='Encoder_CONV2D_4')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), kernel_initializer=model_init,
               padding='same', name='Encoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(256, (3, 3), kernel_initializer=model_init,
               padding='same', name='Encoder_CONV2D_6')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (6, 10, 128)
    x = BatchNormalization()(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), kernel_initializer=model_init, padding='same',
               name='Decoder_CONV2D_6')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    # x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init, padding='same',
               name='Decoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    # x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same',
               name='Decoder_CONV2D_4')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same',
               name='Decoder_CONV2D_3')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), kernel_initializer=model_init, padding='same',
               name='Decoder_CONV2D_2')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), kernel_initializer=model_init, padding='same',
               name='Decoder_CONV2D_1')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer=model_init, padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder

def runPretrain():
    pretrain_data_dirs = [
        '/content/drive/My Drive/Datasets/nexet/nexet_grey_1_1',
        '/content/drive/My Drive/Datasets/nexet/nexet_grey_1_2',
        '/content/drive/My Drive/Datasets/nexet/nexet_grey_1_3',
        '/content/drive/My Drive/Datasets/nexet/nexet_grey_2_1',
        '/content/drive/My Drive/Datasets/nexet/nexet_grey_2_2',
        '/content/drive/My Drive/Datasets/nexet/nexet_grey_2_3',
        '/content/drive/My Drive/Datasets/nexet/nexet_grey_3_1',
        '/content/drive/My Drive/Datasets/nexet/nexet_grey_3_2',
        '/content/drive/My Drive/Datasets/nexet/nexet_grey_3_3'
    ]

    if not _LOAD_MODEL:
        vae_roader = get_model()
    else:
        print("Loading model ...")
        vae_roader = load_model(_EXISTING_MODEL_FILENAME)
        print("Model loaded.")
    total_dirs = len(pretrain_data_dirs)

    for itr in range(0, _iterations):
        print("\n\n\n%s\nIteration %d of %d" % (str(datetime.datetime.now()), itr + 1, _iterations))
        current_run = 1
        for dir in pretrain_data_dirs:
            print("\nPretrain run %d of %d Pretraining on batched images from %s"
                  % (current_run, total_dirs, dir))
            current_run += 1
            X, Y = load_pretrain_data(dir)
            for i in range(0, _epochs):
                print("\n%s\nSub-epoch %d of %d" % (str(datetime.datetime.now()), i + 1, _epochs))
                vae_roader.fit(X, Y,
                               epochs=_epochs,
                               batch_size=_batch_size,
                               shuffle=True,
                               validation_split=0.01)
            vae_roader.save_weights(_WEIGHTS_DIR + 'vae_roader_pretrain_' + str(itr) + '.h5')
            vae_roader.save(_MODEL_FILENAME)
            print("Save model at %s" % _MODEL_FILENAME)
            del X, Y
            gc.collect()


def runTraining():
    if not _LOAD_MODEL:
        vae_roader = get_model()
    else:
        # print("Loading model with weights from %s" % _WEIGHTS_FILE)
        vae_roader = load_model(_EXISTING_MODEL_FILENAME)
        # vae_roader = get_model()
        # vae_roader.load_weights(_WEIGHTS_FILE)
        print("Model loaded.")

    X, Y = load_data(folder=_TRAIN_DATA_DIR)
    for i in range(0, _iterations):
        print("\n\n\n%s\nIteration %d of %d" % (str(datetime.datetime.now()), i + 1, _iterations))
        vae_roader.fit(X, Y,
                       epochs=_epochs,
                       batch_size=_batch_size,
                       shuffle=True,
                       validation_split=0.1,
                       verbose=2)
        vae_roader.save_weights(_WEIGHTS_DIR + 'vae_roader_train_' + str(i) + '.h5')
        vae_roader.save(_MODEL_FILENAME)

    print("\n\n\n%s\nTraining completed." % str(datetime.datetime.now()))

def run_training_on_large_textures(folder, resolution=(1, 1), epochs=1, batch_size=32, iterations=3):
    if not _LOAD_MODEL:
        vae_roader = get_vae_model()
    else:
        vae_roader = load_model(_EXISTING_MODEL_FILENAME)
        print("Model loaded.")
    textures = []
    for root_back, dirs_back, files_back in os.walk(folder):
        for _file in files_back:
            textures.append(_file)

    for i in range(0, iterations):
        print("\n\n\nIteration %d of %d %s ------------------------------" % (i + 1, iterations, str(datetime.datetime.now())))
        current_run = 1
        mean_losses = []
        mean_accuracies = []
        mean_valaccs = []
        for texture in textures:
            print("\nTrain run %d of %d Training on batched images from %s"
                  % (current_run, len(textures), texture))
            current_run += 1
            X, Y = load_data(single_file=os.path.join(folder, texture))
            result = vae_roader.fit(X, Y,
                                    epochs=epochs,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    validation_split=0.05,
                                    verbose=2)
            mean_losses.append(result.history['loss'][-1])
            mean_accuracies.append(result.history['acc'][-1])
            mean_valaccs.append(result.history['val_acc'][-1])
            del X, Y
            sleep(1)
            gc.collect()
            sleep(1)
        print("\nMean on iter %d - loss: %04f - acc: %04f - val_acc: %04f\n"
              % (i+1, statistics.mean(mean_losses), statistics.mean(mean_accuracies), statistics.mean(mean_valaccs)))
        vae_roader.save_weights(_WEIGHTS_DIR + 'vae_roader_train_' + str(i) + '.h5')
        vae_roader.save(_MODEL_FILENAME)

    print("\n\n\n%s\nTraining completed." % str(datetime.datetime.now()))

if __name__ == "__main__":
    if _PRETRAIN:
        runPretrain()
    else:
        # runTraining()
        run_training_on_large_textures(
            folder='/home/rattus/Projects/PythonNN/datasets/2-TEXTURED',
            epochs=2,
            iterations=5,
            batch_size=32)

