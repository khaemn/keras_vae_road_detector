import numpy as np
from PIL import Image
import os
import cv2
import datetime

'''
    This script batches dataset images to larger packs,
    with given vertcial and horizontal count (8*4 by default)
'''

from dataset_generator import generate_noise

_INPUT_DIR = '/home/rattus/Projects/PythonNN/datasets/1-OUT'
_OUTPUT_DIR = '/home/rattus/Projects/PythonNN/datasets/2-TEXTURED'

_H_COUNT = 50
_V_COUNT = 60
_DATA_BATCH_SIZE = _V_COUNT * _H_COUNT

_IMG_WIDTH = 320 * 2  # 160  # 320
_IMG_HEIGHT = 180  # 90  # 180


def compile_batches(shape=(_H_COUNT, _V_COUNT), input_dir=_INPUT_DIR, output_dir=_OUTPUT_DIR, append_noise=True):
    image_files = []
    (h_count, v_count) = shape
    total_textures = h_count * v_count
    # Scan both dirs
    for root_back, dirs_back, files_back in os.walk(input_dir):
        for _file in files_back:
            image_files.append(_file)

    total_files = len(image_files)
    iteration = 1
    print("%s %d files found for processing" % (str(datetime.datetime.now()), total_files))
    (total_batches, remainder) = divmod(total_files, total_textures)
    print("%s %d files can be used in %d batches of size %d"
          % (str(datetime.datetime.now()), total_batches * total_textures, total_batches, total_textures))
    if remainder > 0:
        if append_noise:
            total_necessary_noises = total_textures - remainder
            print("Appending %d noise files to original data..." % total_necessary_noises)
            generate_noise(out_path=input_dir, filecount=total_necessary_noises)
            print("Noise generation completed, please re-run the script to finish batching.")
            quit()
        else:
            print("%s Warning! Remainder is %d files, that will "
                  "not be processed! Image count is not multiple of batch size."
                  % (str(datetime.datetime.now()), remainder))

    debug_index_set = set()
    for batch_index in range(0, total_batches):
        print('Processing batch', iteration, 'of', total_batches)
        iteration += 1
        batched_image = np.zeros((_IMG_HEIGHT * v_count, _IMG_WIDTH * h_count), dtype='uint8')
        for x_index in range(0, h_count):
            for y_index in range(0, v_count):
                index_in_data_array = (batch_index * total_textures + (y_index * h_count) + x_index)
                img = cv2.imread(os.path.join(input_dir, image_files[index_in_data_array]))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                y_offset = _IMG_HEIGHT * y_index
                x_offset = _IMG_WIDTH * x_index
                batched_image[ y_offset:(y_offset + _IMG_HEIGHT),
                              x_offset:(x_offset + _IMG_WIDTH)] = img
                debug_index_set.add(index_in_data_array)
        output = Image.fromarray(batched_image)
        filename = "batched%dx%d_%d_%s" % (h_count, v_count, batch_index, image_files[batch_index])
        output_path = os.path.join(output_dir, filename)
        output.save(output_path)
    print("Processed %d (%d) images." % (total_files, len(debug_index_set)))


if __name__ == "__main__":
    compile_batches()
    # compile_batches(shape=(6, 6),
    #                 output_dir='/home/rattus/Projects/PythonNN/datasets/1-OUT',
    #                 input_dir='/home/rattus/Projects/PythonNN/datasets/2-TEXTURED',
    #                 append_noise=False)
