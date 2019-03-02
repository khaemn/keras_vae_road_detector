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


def get_model():
    print("Building model ...")
    model_init = "he_uniform"
    leak_alpha = 0.1

    # input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_1')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(768, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (6, 10, 128)

    x = Conv2D(768, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_6')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_4')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_3')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = UpSampling2D((2, 2))(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_2')(x)
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


def get_model():
    print("Building model ...")
    model_init = "he_uniform"
    act = 'elu'
    pad = 'same'

    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format
    x = BatchNormalization()(input_img)
    x = Dropout(0.15)(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, # trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_1')(x)
    x = BatchNormalization()(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_2')(x)
    x = BatchNormalization()(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_3')(x)
    x = BatchNormalization()(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_4')(x)
    x = BatchNormalization()(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_5')(x)
    x = BatchNormalization()(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(512, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_6')(x)
    #x = BatchNormalization()(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    # at this point the representation is (6, 10, 128)

    x = Conv2DTranspose(512, (3, 3), kernel_initializer=model_init, padding=pad,
                        strides=(2, 2), activation=act, name='Decoder_CONV2D_6')(x)
    x = Conv2DTranspose(128, (3, 3), kernel_initializer=model_init, padding=pad,
                        strides=(2, 2), activation=act, name='Decoder_CONV2D_5')(x)
    x = Conv2DTranspose(32, (3, 3), kernel_initializer=model_init, padding=pad,
                        strides=(2, 2), activation=act, name='Decoder_CONV2D_4')(x)
    x = Conv2DTranspose(16, (3, 3), kernel_initializer=model_init, padding=pad,
                        strides=(2, 2), activation=act, name='Decoder_CONV2D_3')(x)
    x = Conv2DTranspose(16, (3, 3), kernel_initializer=model_init, padding=pad,
                        strides=(2, 2), activation=act, name='Decoder_CONV2D_2')(x)
    x = Conv2DTranspose(16, (3, 3), kernel_initializer=model_init, padding=pad,
                        strides=(2, 2), activation=act, name='Decoder_CONV2D_1')(x)
    decoded = Conv2DTranspose(1, (3, 3), activation='sigmoid', kernel_initializer=model_init, padding=pad)(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder



def get_dense_model():
    print("Building model ...")
    model_init = "he_uniform"
    act = 'relu'
    pad = 'same'

    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format
    x = BatchNormalization()(input_img)
    x = Dropout(0.15)(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_1')(x)
    x = BatchNormalization()(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_2')(x)
    x = BatchNormalization()(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_3')(x)
    x = BatchNormalization()(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_4')(x)
    x = BatchNormalization()(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_5')(x)
    x = BatchNormalization()(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_6')(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_7')(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 2), padding=pad)(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding=pad, name='Encoder_CONV2D_8')(x)
    x = Activation(activation=act)(x)
    x = MaxPooling2D((2, 3), padding=pad)(x)

    x = Flatten()(x)
    x = Dense(int(img_width * img_height), activation='sigmoid')(x)


    #x = Reshape((int(img_height / 4), int(img_width / 4), 1))(x)
    decoded = Reshape((img_height, img_width, 1))(x)

    # decoded = UpSampling2D((4, 4), interpolation='bilinear')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder



def get_virgos_model():
    pool_size = (2, 2)
    pad = 'valid'
    act = 'relu'

    model = Sequential()
    model.add(BatchNormalization(input_shape=(img_height, img_width, 1)))

    model.add(Conv2D(8, (3, 3), padding=pad, activation=act, name = 'Conv1'))
    model.add(Conv2D(8, (3, 3), padding=pad, activation=act, name = 'Conv2'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(16, (3, 3), padding=pad, activation=act, name = 'Conv3'))
    model.add(Conv2D(16, (3, 3), padding=pad, activation=act, name = 'Conv4'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(32, (3, 3), padding=pad, activation=act, name = 'Conv5'))
    model.add(Conv2D(32, (3, 3), padding=pad, activation=act, name = 'Conv6'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding=pad, activation=act, name = 'Conv7'))
    model.add(Conv2D(64, (3, 3), padding=pad, activation=act, name = 'Conv8'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.2))

    # -------- middle

    model.add(UpSampling2D(size=pool_size))
    model.add(Dropout(0.2))
    model.add(Conv2D(64, (3, 3), padding=pad, activation=act, name = 'Deconv1'))
    model.add(Conv2D(64, (3, 3), padding=pad, activation=act, name = 'Deconv2'))

    model.add(UpSampling2D(size=pool_size))
    model.add(Dropout(0.2))
    model.add(Conv2D(32, (3, 3), padding=pad, activation=act, name = 'Deconv3'))
    model.add(Conv2D(32, (3, 3), padding=pad, activation=act, name = 'Deconv4'))

    model.add(UpSampling2D(size=pool_size))
    model.add(Conv2D(16, (3, 3), padding=pad, activation=act, name = 'Deconv5'))
    model.add(Conv2D(16, (3, 3), padding=pad, activation=act, name = 'Deconv6'))

    model.add(UpSampling2D(size=pool_size))
    model.add(Conv2D(8, (3, 3), padding=pad, activation=act, name = 'Deconv7'))
    model.add(Conv2D(8, (3, 3), padding=pad, activation=act, name = 'Deconv8'))

    # Final layer - only including one channel so 1 filter
    model.add(Conv2D(1, (3, 3), padding=pad, activation=act, name = 'Final'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    model.summary()
    return model

def get_virgos_model():
    pool_size = (2, 2)
    pad = 'same'
    act = 'relu'

    model = Sequential()
    model.add(BatchNormalization(input_shape=(img_height, img_width, 1)))
    model.add(Dropout(0.1))

    model.add(Conv2D(8, (3, 3), padding=pad, activation=act, name = 'Conv1'))
    model.add(Conv2D(8, (3, 3), padding=pad, activation=act, name = 'Conv2'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(16, (3, 3), padding=pad, activation=act, name = 'Conv3'))
    model.add(Conv2D(16, (3, 3), padding=pad, activation=act, name = 'Conv4'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3, 3), padding=pad, activation=act, name = 'Conv5'))
    model.add(Conv2D(32, (3, 3), padding=pad, activation=act, name = 'Conv6'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(64, (3, 3), padding=pad, activation=act, name = 'Conv7'))
    model.add(Conv2D(64, (3, 3), padding=pad, activation=act, name = 'Conv8'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(128, (3, 3), padding=pad, activation=act, name = 'Conv9'))
    model.add(Conv2D(128, (3, 3), padding=pad, activation=act, name = 'Conv10'))
    model.add(MaxPooling2D(pool_size=pool_size))

    # -------- middle
    model.add(UpSampling2D(size=pool_size))
    model.add(Conv2DTranspose(128, (3, 3), padding=pad, activation=act, name = 'Deconv01'))
    model.add(Conv2DTranspose(128, (3, 3), padding=pad, activation=act, name = 'Deconv02'))

    model.add(UpSampling2D(size=pool_size))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, (3, 3), padding=pad, activation=act, name = 'Deconv1'))
    model.add(Conv2DTranspose(64, (3, 3), padding=pad, activation=act, name = 'Deconv2'))

    model.add(UpSampling2D(size=pool_size))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32, (3, 3), padding=pad, activation=act, name = 'Deconv3'))
    model.add(Conv2DTranspose(32, (3, 3), padding=pad, activation=act, name = 'Deconv4'))

    model.add(UpSampling2D(size=pool_size))
    model.add(Conv2DTranspose(16, (3, 3), padding=pad, activation=act, name = 'Deconv5'))
    model.add(Conv2DTranspose(16, (3, 3), padding=pad, activation=act, name = 'Deconv6'))

    model.add(UpSampling2D(size=pool_size))
    model.add(Conv2DTranspose(8, (3, 3), padding=pad, activation=act, name = 'Deconv7'))
    model.add(Conv2DTranspose(8, (3, 3), padding=pad, activation=act, name = 'Deconv8'))

    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(1, (3, 3), padding=pad, activation='sigmoid', name = 'Final'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    model.summary()
    return model



def get_model():
    print("Building model ...")
    model_init = "he_uniform"
    leak_alpha = 0.1

    # input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_1')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(256, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (6, 10, 128)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(256, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_6')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_4')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_3')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_2')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_1')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer=model_init, padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder


def get_virgos_model():
    pool_size = (2, 2)
    pad = 'same'
    act = 'relu'

    model = Sequential()
    model.add(BatchNormalization(input_shape=(img_height, img_width, 1)))
    model.add(Dropout(0.1))

    model.add(Conv2D(8, (3, 3), padding=pad, activation=act, name = 'Conv1'))
    model.add(Conv2D(8, (3, 3), padding=pad, activation=act, name = 'Conv2'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(Conv2D(16, (3, 3), padding=pad, activation=act, name = 'Conv3'))
    model.add(Conv2D(16, (3, 3), padding=pad, activation=act, name = 'Conv4'))
    model.add(MaxPooling2D(pool_size=pool_size))
    model.add(Dropout(0.1))

    model.add(Conv2D(32, (3, 3), padding=pad, activation=act, name = 'Conv5'))
    model.add(Conv2D(32, (3, 3), padding=pad, activation=act, name = 'Conv6'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(64, (3, 3), padding=pad, activation=act, name = 'Conv7'))
    model.add(Conv2D(64, (3, 3), padding=pad, activation=act, name = 'Conv8'))
    model.add(MaxPooling2D(pool_size=pool_size))

    model.add(Conv2D(128, (3, 3), padding=pad, activation=act, name = 'Conv9'))
    model.add(Conv2D(128, (3, 3), padding=pad, activation=act, name = 'Conv10'))
    model.add(MaxPooling2D(pool_size=pool_size))

    # -------- middle
    model.add(UpSampling2D(size=pool_size))
    model.add(Conv2DTranspose(128, (3, 3), padding=pad, activation=act, name = 'Deconv01'))
    model.add(Conv2DTranspose(128, (3, 3), padding=pad, activation=act, name = 'Deconv02'))

    model.add(UpSampling2D(size=pool_size))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(64, (3, 3), padding=pad, activation=act, name = 'Deconv1'))
    model.add(Conv2DTranspose(64, (3, 3), padding=pad, activation=act, name = 'Deconv2'))

    model.add(UpSampling2D(size=pool_size))
    model.add(BatchNormalization())
    model.add(Conv2DTranspose(32, (3, 3), padding=pad, activation=act, name = 'Deconv3'))
    model.add(Conv2DTranspose(32, (3, 3), padding=pad, activation=act, name = 'Deconv4'))

    model.add(UpSampling2D(size=pool_size))
    model.add(Conv2DTranspose(16, (3, 3), padding=pad, activation=act, name = 'Deconv5'))
    model.add(Conv2DTranspose(16, (3, 3), padding=pad, activation=act, name = 'Deconv6'))

    model.add(UpSampling2D(size=pool_size))
    model.add(Conv2DTranspose(8, (3, 3), padding=pad, activation=act, name = 'Deconv7'))
    model.add(Conv2DTranspose(8, (3, 3), padding=pad, activation=act, name = 'Deconv8'))

    # Final layer - only including one channel so 1 filter
    model.add(Conv2DTranspose(1, (3, 3), padding=pad, activation='sigmoid', name = 'Final'))

    model.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    model.summary()
    return model


def get_model():
    print("Building model ...")
    model_init = "he_uniform"
    leak_alpha = 0.1
    dropout = 0.1

    # input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(8, (3, 3), kernel_initializer=model_init, # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_1')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(384, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (6, 10, 128)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(384, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_6')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_4')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_3')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_2')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_1')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer=model_init, padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder

def get_mini_model():
    print("Building model ...")
    model_init = "he_uniform"
    leak_alpha = 0.1
    dropout = 0.1

    # input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(4, (3, 3), kernel_initializer=model_init, # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_1')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(4, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4'
              , activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5'
              , activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(128, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6'
              , activation='relu')(x)
    # x = BatchNormalization()(x)
    # x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (6, 10, 128)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_6')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_4')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_3')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_2')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(4, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_1')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer=model_init, padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder


def get_pipe_model():
    print("Building model ...")
    model_init = "he_uniform"
    leak_alpha = 0.1
    dropout = 0.1

    # input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format

    x = Conv2D(16, (3, 3), kernel_initializer=model_init, # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_1')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4a'
              , activation='relu')(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4b'
              , activation='relu')(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4c'
              , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5a'
              , activation='relu')(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5b'
              , activation='relu')(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5c'
              , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6a'
              , activation='relu')(x)
    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6b'
              , activation='relu')(x)
    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  #trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6c'
              , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (6, 10, 128)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_6')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_4')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_3')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(4, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_2')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(4, (3, 3), kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_1')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer=model_init, padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder

def get_micro2_model():
    print("Building model ...")
    model_init = "he_uniform"
    leak_alpha = 0.05
    dropout = 0.15

    # input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format

    x = BatchNormalization()(input_img)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_1'
               , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_2'
               , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = BatchNormalization()(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_3'
               , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = BatchNormalization()(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4'
               , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(64, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5'
               , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = BatchNormalization()(x)
    x = Conv2D(128, (3, 3), kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6'
               , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (6, 10, 128)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), kernel_initializer=model_init, padding='same',
               activation='relu', name='Decoder_CONV2D_6')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), kernel_initializer=model_init, padding='same',
               activation='relu', name='Decoder_CONV2D_5')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same',
               activation='relu', name='Decoder_CONV2D_4')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same',
               activation='relu', name='Decoder_CONV2D_3')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same',
               activation='relu', name='Decoder_CONV2D_2')(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(16, (3, 3), kernel_initializer=model_init, padding='same',
               activation='relu', name='Decoder_CONV2D_1')(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', kernel_initializer=model_init, padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder


def get_pipe_model():
    print("Building model ...")
    model_init = "he_uniform"
    leak_alpha = 0.1
    dropout = 0.1

    # input_img = Input(shape=(img_height, img_width, 3))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(img_height, img_width, 1))  # adapt this if using `channels_first` image data format
    kernel = (3, 3)
    x = Conv2D(16, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_1')(input_img)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(16, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_2')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(16, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_3')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=leak_alpha)(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Dropout(dropout)(x)

    x = Conv2D(32, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4a'
               , activation='relu')(x)
    x = Conv2D(32, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4b'
               , activation='relu')(x)
    x = Conv2D(32, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_4c'
               , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(32, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5a'
               , activation='relu')(x)
    x = Conv2D(32, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5b'
               , activation='relu')(x)
    x = Conv2D(32, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_5c'
               , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    x = Conv2D(96, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6a'
               , activation='relu')(x)
    x = Conv2D(96, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6b'
               , activation='relu')(x)
    x = Conv2D(96, kernel, kernel_initializer=model_init,  # trainable=_PRETRAIN,
               padding='same', name='Encoder_CONV2D_6c'
               , activation='relu')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (6, 10, 128)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(96, kernel, kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_6')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, kernel, kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_5')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, kernel, kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_4')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(8, kernel, kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_3')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(4, kernel, kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_2')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    x = UpSampling2D((2, 2))(x)
    x = Conv2D(4, kernel, kernel_initializer=model_init, padding='same', name='Decoder_CONV2D_1')(x)
    x = LeakyReLU(alpha=leak_alpha)(x)

    decoded = Conv2D(1, kernel, activation='sigmoid', kernel_initializer=model_init, padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy',
                        metrics=['accuracy'])
    print("Built model info:\n")
    autoencoder.summary()
    return autoencoder