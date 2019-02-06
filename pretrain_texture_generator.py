import numpy as np
from PIL import Image
import os
import cv2
import datetime

'''
    This script takes JPG images of any resolution, converts them to grayscale,
    resizes to VAE_ROADER input size (320*180 by default) and packs batches of
    images to large textures, using 40*40 resized images by default.
    This approach significantly reduces IO operations on cloud server while pretrain
    of VAE_ROADER.
    As far as this is pretrain images, no segmentation masks are processed.    
'''

_INPUT_DIR = 'D:/__PROJECTS/PythonNN/datasets/NEXET/nexet_2017_train_3/nexet_2017_3'
_OUTPUT_DIR = 'data/nexet_grey_3'

# Usual resolutuion of HD cam is 1280*720, we use /4 resolution here to save performance
_X_WIDTH = 320  # 160  # 320
_X_HEIGHT = 180  # 90  # 180

# Texture size
_H_COUNT = 40
_V_COUNT = 40
_DATA_BATCH_SIZE = _V_COUNT * _H_COUNT


def readAsTexturePiece(filepath="", resolution=(_X_WIDTH, _X_HEIGHT)):
    (gen_w, gen_h) = resolution
    x_img = cv2.imread(filepath)
    x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)

    x_resized = cv2.resize(x_img, (gen_w, gen_h), interpolation=cv2.INTER_LANCZOS4)
    output = np.array(x_resized, dtype='uint8')
    return output

def generate_dataset(input_dir=_INPUT_DIR, output_dir=_OUTPUT_DIR, resolution=(_X_WIDTH, _X_HEIGHT)):
    image_files = []

    (gen_w, gen_h) = resolution

    for root_back, dirs_back, files_back in os.walk(_INPUT_DIR):
        for _file in files_back:
            image_files.append(_file)

    total_files = len(image_files)
    iteration = 1
    print("%s %d files found for processing" % (str(datetime.datetime.now()), total_files))
    (total_batches, remainder) = divmod(total_files, _DATA_BATCH_SIZE)
    print("%s %d files can be used in %d batches of size %d"
          % (str(datetime.datetime.now()), total_batches * _DATA_BATCH_SIZE, total_batches, _DATA_BATCH_SIZE))
    if remainder > 0:
        print("%s Warning! Remainder is %d files, that will "
              "not be processed! Image count is not multiple of batch size."
              % (str(datetime.datetime.now()), remainder))

    debug_index_set = set()
    for batch_index in range(0, total_batches):
        print('Processing batch', iteration, 'of', total_batches)
        iteration += 1
        batched_image = np.zeros((_X_HEIGHT * _V_COUNT, _X_WIDTH * _H_COUNT), dtype='uint8')
        for x_index in range(0, _H_COUNT):
            for y_index in range(0, _V_COUNT):
                index_in_data_array = (batch_index * _DATA_BATCH_SIZE + (y_index * _H_COUNT) + x_index)
                img = readAsTexturePiece(os.path.join(input_dir, image_files[index_in_data_array])
                                         , resolution)
                y_offset = _X_HEIGHT * y_index
                x_offset = _X_WIDTH * x_index
                batched_image[y_offset:(y_offset + _X_HEIGHT),
                x_offset:(x_offset + _X_WIDTH)] = img
                debug_index_set.add(index_in_data_array)
        output = Image.fromarray(batched_image)
        filename = "batch%d_%s" % (batch_index, image_files[batch_index])
        output_path = os.path.join(_OUTPUT_DIR, filename)
        output.save(output_path)
    print("Processed %d (%d) images." % (total_files, len(debug_index_set)))


if __name__ == '__main__':
    generate_dataset(_INPUT_DIR, _OUTPUT_DIR)