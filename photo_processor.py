from keras.models import Sequential, load_model
import numpy as np
from PIL import Image
import os
import cv2
from video_processor import RoadDetector

_MODEL_FILENAME = 'models/ext_model_yolike_roader.h5'
_INPUT_DIR = '/home/rattus/Projects/PythonNN/datasets/nexet-1/images/'
_OUT_DIR = '/home/rattus/Projects/PythonNN/datasets/nexet-1/gen_masks/'
_SHOW = False

def processPhotos(input_dir=_INPUT_DIR, out_dir=_OUT_DIR):
    detector = RoadDetector(_MODEL_FILENAME)
    files = []

    for root_back, dirs_back, files_back in os.walk(input_dir):
        for _file in files_back:
            files.append(_file)

    total_files = len(files)
    iteration = 1
    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

    for filename in files:
        print('Processing file', iteration, 'of', total_files, filename)
        iteration += 1

        x_img = cv2.imread(os.path.join(input_dir, filename))
        if _SHOW:
            cv2.imshow("Original", x_img)
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)
        (origin_h, origin_w) = x_img.shape

        nn_output = detector.predict(x_img)
        if _SHOW:
            cv2.imshow("Prediction", nn_output)

        nn_output = cv2.resize(nn_output, (origin_w, origin_h), interpolation=cv2.INTER_LANCZOS4)
        masking_threshold = 80
        masking_max = detector.max_RGB
        _, mask = cv2.threshold(nn_output,
                                masking_threshold,
                                masking_max,
                                cv2.THRESH_BINARY)

        # Preprocess to reduce noise
        preprocessing_iter = 2
        mask = cv2.dilate(cv2.erode(mask, big_kernel, iterations=preprocessing_iter),
                          small_kernel, iterations=preprocessing_iter)
        mask = cv2.dilate(cv2.erode(mask, big_kernel, iterations=preprocessing_iter),
                          small_kernel, iterations=preprocessing_iter)

        mask = mask.astype(np.uint8)

        out_img = Image.fromarray(mask)
        output_path = os.path.join(out_dir, filename)
        out_img.save(output_path)

        if _SHOW:
            cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    dirs = [
            # '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/nexet3-day-part2',
            # '/home/rattus/Projects/PythonNN/datasets/downloaded-assorted',
            # '/home/rattus/Projects/PythonNN/datasets/nexet_example',
            # '/home/rattus/Projects/PythonNN/datasets/noroad-maskeds',
            # '/home/rattus/Projects/PythonNN/datasets/road2and5-maskeds',
            # '/home/rattus/Projects/PythonNN/datasets/road3-maskeds',
            # '/home/rattus/Projects/PythonNN/datasets/road4-maskeds',
            # '/home/rattus/Projects/PythonNN/datasets/road6-maskeds',
            # '/home/rattus/Projects/PythonNN/datasets/road6-reduced',
            # '/home/rattus/Projects/PythonNN/datasets/road8-maskeds',
            # '/home/rattus/Projects/PythonNN/datasets/road9-maskeds',
            '/home/rattus/Projects/PythonNN/datasets/road10-maskeds',
            # '/home/rattus/Projects/PythonNN/datasets/road71-maskeds',
            # '/home/rattus/Projects/PythonNN/datasets/road-4-12-15-gen'

            # '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fvid/from_vid_5',
            # '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fvid/from_vid_6',
            # '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fvid/from_vid_7',
            # 'dataset/sim'
            # '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fkievvid',
           ]
    #out_dir = '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fkievvid'
    # out_dir = '/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/nexet3-day-part2'
    out_dir = '/home/rattus/Projects/PythonNN/datasets/road10-maskeds'
    for d in dirs:
        path = os.path.join(d, 'images')
        out_path = os.path.join(d, 'gen-masks')
        print("Processing directory %s" % path)
        processPhotos(path, out_path)
