from vis.visualization import visualize_activation
from vis.utils import utils
from keras import activations
import numpy as np
from keras.models import load_model

from matplotlib import pyplot as plt

plt.rcParams['figure.figsize'] = (18, 6)

_MODEL_FILENAME = 'models/vaed_model_yolike_roader.h5'

model = load_model(_MODEL_FILENAME)
from vis.visualization import get_num_filters

# The name of the layer we want to visualize
# You can see this in the model definition.
layer_name = 'Encoder_CONV2D_3'
layer_idx = utils.find_layer_idx(model, layer_name)

# Visualize all filters in this layer.
filters = np.arange(get_num_filters(model.layers[layer_idx]))

# Generate input image for each filter.
vis_images = []
for idx in filters:
    img = visualize_activation(model, layer_idx, filter_indices=idx)

    # Utility to overlay text on image.
    img = utils.draw_text(img, 'Filter {}'.format(idx))
    vis_images.append(img)

# Generate stitched image palette with 8 cols.
stitched = utils.stitch_images(vis_images, cols=8)
plt.axis('off')
plt.imshow(stitched)
plt.title(layer_name)
plt.show()
plt.imsave('/filters/filt.png')
