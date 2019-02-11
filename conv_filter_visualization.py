#  https://github.com/keras-team/keras/blob/master/examples/conv_filter_visualization.py
'''Visualization of the filters of VGG16, via gradient ascent in input space.
This script can run on CPU in a few minutes.
Results example: http://i.imgur.com/4nj4KjN.jpg
'''
from __future__ import print_function

import numpy as np
import time
from keras.preprocessing.image import save_img
from keras import backend as K
from keras.models import load_model
import math


# util function to convert a tensor into a valid image
def deprocess_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + K.epsilon())
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_data_format() == 'channels_first':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + K.epsilon())

# dimensions of the generated pictures for each filter.
img_width = 256  # 128
img_height = img_width # 256  # 128
channel_count = 1  # 1 for grey, 3 for rgb


_MODEL_FILENAME = 'models/vae_model_yolike_roader.h5'
model = load_model(_MODEL_FILENAME)

print('Model loaded.')

model.summary()
# this is the placeholder for the input images

input_img = model.input
# get the symbolic outputs of each "key" layer (we gave them unique names).

layer_dict = dict([(layer.name, layer) for layer in model.layers[0:]])

def plotFiltersFor(layer_name, filter_count=8):
    kept_filters = []
    total_filters = filter_count
    gradient_steps = 100

    # we start from a gray image with some random noise
    if K.image_data_format() == 'channels_first':
        origin_input_img_data = np.random.random((1, channel_count, img_width, img_height))
    else:
        origin_input_img_data = np.random.random((1, img_width, img_height, channel_count))
    origin_input_img_data = (origin_input_img_data - 0.5) * 20 + 128

    for filter_index in range(total_filters):  # number of filters in the conv layer
        print('Processing filter %d of %d' % (filter_index, total_filters))
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        layer_output = layer_dict[layer_name].output
        if K.image_data_format() == 'channels_first':
            loss = K.mean(layer_output[:, filter_index, :, :])
        else:
            loss = K.mean(layer_output[:, :, :, filter_index])

        print('Getting gradients...')
        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_img], [loss, grads])

        # step size for gradient ascent
        step = 1.

        print('Filling randoms...')

        input_img_data = origin_input_img_data.copy()
        print('Calculating gradients...')
        # we run gradient ascent for N steps
        for i in range(gradient_steps):
            loss_value, grads_value = iterate([input_img_data])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)
            if loss_value <= 0.:
                # some filters get stuck to 0, we can skip them
                break

        # decode the resulting input image
        if loss_value > 0:
            print('Deprocessing image...')
            img = deprocess_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # we will stitch the best 64 filters on a 8 x 8 grid.
    n = max(1, int(math.sqrt(abs(len(kept_filters) - 1))))

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for
    # our 8 x 8 filters of size 128 x 128, with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, 3))

    print("Kept filters:", len(kept_filters))
    if len(kept_filters) > 0:
        # fill the picture with our saved filters
        for i in range(n):
            for j in range(n):
                _index = i * n + j
                img, loss = kept_filters[_index]
                width_margin = (img_width + margin) * i
                height_margin = (img_height + margin) * j
                stitched_filters[
                    width_margin: width_margin + img_width,
                    height_margin: height_margin + img_height, :] = img

    # save the result to disk
    save_img('filters/' + layer_name + ('_stitched_filters_%dx%d.png' % (n, n)), stitched_filters)

# the name of the layer we want to visualize
# (see model definition at keras/applications/vgg16.py)
layer_names = [ #['Encoder_CONV2D_1', 8],
                #['Encoder_CONV2D_2', 16],
                #['Encoder_CONV2D_3', 32],
                #['Encoder_CONV2D_4', 64],
                #['Encoder_CONV2D_5', 128],
                ['Encoder_CONV2D_6', 32],
               ]

for lname in layer_names:
    plotFiltersFor(lname[0], lname[1])

