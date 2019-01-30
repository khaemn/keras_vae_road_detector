import numpy as np
from PIL import Image
import os
import cv2
import datetime

'''
    This script batches dataset images to larger packs,
    with given vertcial and horizontal count (8*4 by default)
'''

_INPUT_DIR = 'dataset/train/X'
_OUTPUT_DIR = ''

_V_COUNT = 8
_H_COUNT = 4
_BATCH_SIZE = _V_COUNT * _H_COUNT

_IMG_WIDTH = 320 * 2  # 160  # 320
_IMG_HEIGHT = 180  # 90  # 180


def compile_batches(input_dir=_INPUT_DIR, output_dir=_OUTPUT_DIR):
    image_files = []

    # Scan both dirs
    for root_back, dirs_back, files_back in os.walk(input_dir):
        for _file in files_back:
            image_files.append(_file)

    total_files = len(image_files)
    iteration = 1
    print("%s %d files found for processing" % (str(datetime.datetime.now()), total_files))
    (multiple, remainder) = divmod(total_files, _BATCH_SIZE)
    print("%s %d files can be used in %d batches of size %d"
          % (str(datetime.datetime.now()), multiple * _BATCH_SIZE, multiple, _BATCH_SIZE))
    if remainder > 0:
        print("%s Warning! Remainder is %d files, that will "
              "not be processed! Image count is not multiple of batch size."
              % (str(datetime.datetime.now()), remainder))
    quit()

    for filename in image_files:
        print('Processing file', iteration, 'of', total_files)
        iteration += 1
        img = cv2.imread(os.path.join(input_dir, filename))


    return

if __name__ == "__main__":
    compile_batches()
