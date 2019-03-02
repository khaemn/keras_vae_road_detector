import numpy as np
from PIL import Image
import os
import cv2
import datetime

'''
    This script takes 2 folders with images and forms a combined image for each pair
    also performing scaling to given shape.
    By default, VAERoader takes 320*180 pixel image as input and provides the same size
    output.
    Heatmapped images usually are about 1280*720 (full-hd camera) images, heatmapped by OCV_RND project.
'''

_DATA_DIR = '/home/rattus/Projects/PythonNN/datasets/diy-road-photos/'
_X_INPUT_DIR = _DATA_DIR + 'images-expanded'
_Y_INPUT_DIR = _DATA_DIR + 'masks-expanded'
_DATASET_DIR = 'dataset'
_TRAIN_DIR = os.path.join(_DATASET_DIR, 'train')
_OUT_DIR = os.path.join(_TRAIN_DIR, 'X')

_FLIP_HALVES = False

# Usual resolution of HD cam is 1280*720, we use /4 resolution here to save performance
_X_WIDTH = 320  # 160  # 320
_X_HEIGHT = 180  # 90  # 180

_Y_WIDTH = _X_WIDTH
_Y_HEIGHT = _X_HEIGHT


def generate_from_dirs(in_dirs, out_dir, batch_dims=(1,1)):
    total_dirs = len(in_dirs)
    dir_num = 1
    start = datetime.datetime.now()
    total_files = 0
    for dir in in_dirs:
        print("Processing dir %s (%d of %d) %s"
              % (dir, dir_num, total_dirs, datetime.datetime.now()))
        dir_num += 1
        x_inp = os.path.join(dir, 'images')
        y_inp = os.path.join(dir, 'masks')
        total_files += generate_dataset(x_input=x_inp,
                                        y_input=y_inp,
                                        out_path=out_dir,
                                        augmenting=True,
                                        batch_dims=(1, 1))
    end = datetime.datetime.now()
    total_time = end - start
    print("Total %d files processed" % total_files)
    print("Elapsed time %s" % total_time)


def generate_noise(xresolution=(_X_WIDTH, _X_HEIGHT), out_path="", filecount=1):

    (gen_w, gen_h) = xresolution
    scale = 12

    for i in range(0, filecount):
        x_data = np.random.random((gen_h // scale, gen_w // scale))
        x_data *= 255
        x_data = x_data.astype('uint8')
        x_data = cv2.resize(x_data, (gen_w, gen_h))
        output = np.zeros((gen_h, gen_w * 2), dtype='uint8')
        output[:, :gen_w] = x_data

        f_output = Image.fromarray(output)
        output_path = os.path.join(out_path, "noised-%d.jpg" % i)
        f_output.save(output_path)
        # print("    Noised output")
    return True

def generate_dataset(resolution=(_X_WIDTH, _X_HEIGHT),
                     x_input=_X_INPUT_DIR,
                     y_input=_Y_INPUT_DIR,
                     out_path=_OUT_DIR,
                     augmenting=False,
                     batch_dims=(1, 1)):
    x_images = []
    y_images = []

    (gen_w, gen_h) = resolution

    # Scan both dirs
    for root_back, dirs_back, files_back in os.walk(x_input):
        for _file in files_back:
            x_images.append(_file)

    for root_back, dirs_back, files_back in os.walk(y_input):
        for _file in files_back:
            y_images.append(_file)

    assert len(x_images) == len(y_images)

    total_files = len(x_images)
    iteration = 1
    (h_count, v_count) = batch_dims
    data_batch_size = h_count * v_count
    print("%s %d files found for processing" % (str(datetime.datetime.now()), total_files))
    (total_batches, remainder) = divmod(total_files, data_batch_size)
    print("%s %d files can be used in %d batches of size %d"
          % (str(datetime.datetime.now()), total_batches * data_batch_size, total_batches, data_batch_size))
    if remainder > 0:
        print("%s Warning! Remainder is %d files, that will "
              "not be processed! Image count is not multiple of batch size."
              % (str(datetime.datetime.now()), remainder))

    for filename in x_images:
        print('Processing file', iteration, 'of', total_files, '  ', filename)
        iteration += 1
        x_img = cv2.imread(os.path.join(x_input, filename))
        convertation_color_space = cv2.COLOR_BGR2GRAY
        x_img = cv2.cvtColor(x_img, convertation_color_space)
        (origin_h, origin_w) = x_img.shape

        y_img = cv2.imread(os.path.join(y_input, filename))
        y_img = cv2.cvtColor(y_img, convertation_color_space)
        y_img = cv2.resize(y_img, (origin_w, origin_h))

        assert x_img.shape == y_img.shape

        x_resized = cv2.resize(x_img, (gen_w, gen_h), interpolation=cv2.INTER_LINEAR)
        # x_resized = cv2.equalizeHist(x_resized)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        x_resized = clahe.apply(x_resized)
        y_resized = cv2.resize(y_img, (gen_w, gen_h), interpolation=cv2.INTER_LINEAR)

        output = np.zeros((gen_h, gen_w * 2), dtype='uint8')
        output[:, :gen_w] = x_resized
        output[:, gen_w:] = y_resized

        f_output = Image.fromarray(output)
        output_path = os.path.join(out_path, filename)
        f_output.save(output_path)
        print("    Regular output")

        _SWAP_HALVES = augmenting
        _FLIP_HORIZ = augmenting
        _SWAP_VERT_HALVES = augmenting
        _EXTRACT_CENTER = augmenting

        if _SWAP_HALVES:
            hor_middle = int(gen_w / 2)

            halved_x = x_resized.copy()
            halved_x[:, :hor_middle] = x_resized[:, hor_middle:]
            halved_x[:, hor_middle:] = x_resized[:, :hor_middle]

            halved_y = y_resized.copy()
            halved_y[:, :hor_middle] = y_resized[:, hor_middle:]
            halved_y[:, hor_middle:] = y_resized[:, :hor_middle]

            halved = np.zeros((gen_h, gen_w * 2), dtype='uint8')
            halved[:, :gen_w] = halved_x
            halved[:, gen_w:] = halved_y
            halved_output = Image.fromarray(halved)
            output_path = os.path.join(out_path, 'halved-' + filename)
            halved_output.save(output_path)
            print("    Horizontal half-swapping output")

        if _SWAP_VERT_HALVES:
            vert_middle = int(gen_h / 2)

            halved_x = x_resized.copy()
            halved_x_src = cv2.flip(halved_x, 0)
            halved_x[:vert_middle, :] = halved_x_src[vert_middle:, :]
            halved_x[vert_middle:, :] = halved_x_src[:vert_middle, :]

            halved_y = y_resized.copy()
            halved_y_src = cv2.flip(halved_y, 0)
            halved_y[:vert_middle, :] = halved_y_src[vert_middle:, :]
            halved_y[vert_middle:, :] = halved_y_src[:vert_middle, :]

            halved = np.zeros((gen_h, gen_w * 2), dtype='uint8')
            halved[:, :gen_w] = halved_x
            halved[:, gen_w:] = halved_y
            halved_output = Image.fromarray(halved)
            output_path = os.path.join(out_path, 'v-swap-' + filename)
            halved_output.save(output_path)
            print("    Vertical half-swapping + flip output")

        if _FLIP_HORIZ:
            hflip_x = cv2.flip(x_resized, 1)
            hflip_y = cv2.flip(y_resized, 1)
            horflipped = np.zeros((gen_h, gen_w * 2), dtype='uint8')
            horflipped[:, :gen_w] = hflip_x
            horflipped[:, gen_w:] = hflip_y
            f_horflipped = Image.fromarray(horflipped)
            output_path = os.path.join(out_path, 'hor-flip-' + filename)
            f_horflipped.save(output_path)
            print("    Horizontal flipping output")

        if _EXTRACT_CENTER:
            extraction_factor = 3
            extraction_w = origin_w // extraction_factor
            extraction_h = origin_h // extraction_factor
            x_offs = (origin_w - extraction_w) // 2
            y_offs = (origin_h - extraction_h) // 2
            centr_x = x_img[y_offs:y_offs+extraction_h, x_offs:x_offs+extraction_w]
            centr_x = cv2.resize(centr_x, (gen_w, gen_h))
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            centr_x = clahe.apply(centr_x)
            centr_y = y_img[y_offs:y_offs + extraction_h, x_offs:x_offs + extraction_w]
            centr_y = cv2.resize(centr_y, (gen_w, gen_h))
            centered = np.zeros((gen_h, gen_w * 2), dtype='uint8')
            centered[:, :gen_w] = centr_x
            centered[:, gen_w:] = centr_y
            f_centered = Image.fromarray(centered)
            output_path = os.path.join(out_path, 'center-' + filename)
            f_centered.save(output_path)
            print("    Extracted center output")
            centr_x = cv2.flip(centr_x, 1)
            centr_y = cv2.flip(centr_y, 1)
            centered[:, :gen_w] = centr_x
            centered[:, gen_w:] = centr_y
            f_centered = Image.fromarray(centered)
            output_path = os.path.join(out_path, 'center-horflip-' + filename)
            f_centered.save(output_path)
            print("    Extracted center horflip output")

    return total_files


if __name__ == '__main__':
    # generate_dataset()

    # generate_noise(out_path='/home/rattus/Projects/PythonNN/datasets/2-TEXTURED',
    #                filecount=5)
    # quit()
    generate_from_dirs(in_dirs=[
                                # '/home/rattus/Projects/PythonNN/datasets/diy-road-photos',
                                # '/home/rattus/Projects/PythonNN/datasets/downloaded-assorted',
                                # '/home/rattus/Projects/PythonNN/datasets/nexet_example',
                                # '/home/rattus/Projects/PythonNN/datasets/noroad-maskeds',
                                # '/home/rattus/Projects/PythonNN/datasets/road2and5-maskeds',
                                # '/home/rattus/Projects/PythonNN/datasets/road3-maskeds',
                                # '/home/rattus/Projects/PythonNN/datasets/road4-maskeds',
                                # '/home/rattus/Projects/PythonNN/datasets/road6-reduced',
                                # '/home/rattus/Projects/PythonNN/datasets/road8-maskeds',
                                # '/home/rattus/Projects/PythonNN/datasets/road9-maskeds',
                                # '/home/rattus/Projects/PythonNN/datasets/road10-maskeds',
                                # '/home/rattus/Projects/PythonNN/datasets/road71-maskeds',
                                # '/home/rattus/Projects/PythonNN/datasets/road-4-12-15-gen',
                                # '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fvid/from_vid_1',
                                # '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fvid/from_vid_2',
                                # '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fvid/from_vid_3',
                                # '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fvid/from_vid_4',
                                # '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fvid/from_vid_5',
                                # '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fvid/from_vid_7',
                                '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/nexet3-day-part1',
                                '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/nexet3-day-part3',

                                # Test!!!;
                                # '/home/rattus/Projects/PythonNN/datasets/downloaded-assorted'
                                #'/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fkievvid',
                               ],
                       out_dir='/home/rattus/Projects/PythonNN/datasets/1-OUT')
                       # out_dir = '/home/rattus/Projects/PythonNN/datasets/3-TEST')
                    # out_dir='/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/nexet3-day-part3')
                       # out_dir = '/home/rattus/Projects/PythonNN/datasets/downloaded-assorted/data')


