import numpy as np
from PIL import Image
import os
import cv2

'''
    This script takes 2 folders with images and forms a combined image for each pair
    also performing scaling to given shape.
    By default, VAERoader takes 320*180 pixel image as input and provides the same size
    output.
    Heatmapped images usually are about 1280*720 (full-hd camera) images, heatmapped by OCV_RND project.
'''

# _X_INPUT_DIR = 'heatmapping/heatmap_src'
_DATA_DIR = '/home/rattus/Projects/PythonNN/datasets/diy-road-photos/'
_X_INPUT_DIR = _DATA_DIR + 'images-expanded'
# _Y_INPUT_DIR = 'heatmapping/heatmap_out'
_Y_INPUT_DIR = _DATA_DIR + 'masks-expanded'
_DATASET_DIR = 'dataset'
_TRAIN_DIR = os.path.join(_DATASET_DIR, 'train')
_X_TRAIN_DIR = os.path.join(_TRAIN_DIR, 'X')
_Y_TRAIN_DIR = os.path.join(_TRAIN_DIR, 'Y')

_ATTACH_Y_TO_X = True
_CONVERT_TO_GRAYSCALE = True
_FLIP_HALVES = False

# Usual resolutuion of HD cam is 1280*720, we use /4 resolution here to save performance
_X_WIDTH = 320  # 160  # 320
_X_HEIGHT = 180  # 90  # 180

# I think that road heatmap resolution of R/20 will be enough
# _Y_WIDTH = 64
# _Y_HEIGHT = 36
_Y_WIDTH = _X_WIDTH
_Y_HEIGHT = _X_HEIGHT


def generate_dataset(resolution=(_X_WIDTH, _X_HEIGHT)):
    x_images = []
    y_images = []

    (gen_w, gen_h) = resolution

    # Scan both dirs
    for root_back, dirs_back, files_back in os.walk(_X_INPUT_DIR):
        for _file in files_back:
            x_images.append(_file)

    for root_back, dirs_back, files_back in os.walk(_Y_INPUT_DIR):
        for _file in files_back:
            y_images.append(_file)

    assert len(x_images) == len(y_images)

    total_files = len(x_images)
    iteration = 1

    for filename in x_images:
        print('Processing file', iteration, 'of', total_files)
        iteration += 1
        x_img = cv2.imread(os.path.join(_X_INPUT_DIR, filename))
        convertation_color_space = cv2.COLOR_BGR2GRAY if _CONVERT_TO_GRAYSCALE else cv2.COLOR_BGR2RGB
        x_img = cv2.cvtColor(x_img, convertation_color_space)
        (origin_h, origin_w) = x_img.shape

        y_img = cv2.imread(os.path.join(_Y_INPUT_DIR, filename))
        y_img = cv2.cvtColor(y_img, convertation_color_space)
        y_img = cv2.resize(y_img, (origin_w, origin_h))

        assert x_img.shape == y_img.shape

        depth = 1
        if _CONVERT_TO_GRAYSCALE:
            (height, width) = x_img.shape
        else:
            (height, width, depth) = x_img.shape

        x_resized = cv2.resize(x_img, (gen_w, gen_h), interpolation=cv2.INTER_LINEAR)
        y_resized = cv2.resize(y_img, (gen_w, gen_h), interpolation=cv2.INTER_LINEAR)

        _FLIP_HALVES = (iteration % 5 == 0)

        if _FLIP_HALVES:
            hor_middle = int(gen_w / 2)

            flipped_x = x_resized.copy()
            flipped_x[:, :hor_middle] = x_resized[:, hor_middle:]
            flipped_x[:, hor_middle:] = x_resized[:, :hor_middle]
            x_resized = flipped_x

            flipped_y = y_resized.copy()
            flipped_y[:, :hor_middle] = y_resized[:, hor_middle:]
            flipped_y[:, hor_middle:] = y_resized[:, :hor_middle]
            y_resized = flipped_y

        if _ATTACH_Y_TO_X:
            # Concatenating input with expected output
            if _CONVERT_TO_GRAYSCALE:
                output = np.zeros((gen_h, gen_w * 2), dtype='uint8')
                output[:, :_X_WIDTH] = x_resized
                output[:, _X_WIDTH:] = y_resized
            else:
                output = np.zeros((gen_h, gen_w*2, depth), dtype='uint8')
                output[:, :_X_WIDTH] = x_resized
                output[:, _X_WIDTH:] = y_resized

            output = Image.fromarray(output)
            output_path = os.path.join(_X_TRAIN_DIR, filename)
            if _FLIP_HALVES:
                output_path = os.path.join(_X_TRAIN_DIR, 'halved-' + filename)
            output.save(output_path)
        else:
            x_out, y_out = Image.fromarray(x_resized), Image.fromarray(y_resized)
            x_out.save(os.path.join(_X_TRAIN_DIR, filename))
            y_out.save(os.path.join(_Y_TRAIN_DIR, filename))


if __name__ == '__main__':
    generate_dataset()