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