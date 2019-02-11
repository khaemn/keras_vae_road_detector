from keras.models import Sequential, load_model
import numpy as np
from PIL import Image
import os
import cv2
from video_processor import RoadDetector

_MODEL_FILENAME = 'models/vae_model_yolike_roader.h5'
_INPUT_DIR = '/home/rattus/Projects/PythonNN/datasets/nexet_example/images/'
# '/home/rattus/Projects/PythonNN/datasets/nexet_example/smalltest/'
_OUT_DIR = '/home/rattus/Projects/PythonNN/datasets/nexet_example/gen_masks/'
# '/home/rattus/Projects/PythonNN/datasets/nexet_example/gen_masks/'
_SHOW = False

def processPhotos(input_dir=_INPUT_DIR, out_dir=_OUT_DIR):
    detector = RoadDetector(_MODEL_FILENAME)
    files = []

    for root_back, dirs_back, files_back in os.walk(input_dir):
        for _file in files_back:
            files.append(_file)

    total_files = len(files)
    iteration = 1

    for filename in files:
        print('Processing file', iteration, 'of', total_files)
        iteration += 1

        x_img = cv2.imread(os.path.join(_INPUT_DIR, filename))
        if _SHOW:
            cv2.imshow("Original", x_img)
        x_img = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)
        (origin_h, origin_w) = x_img.shape

        nn_output = detector.predict(x_img)
        if _SHOW:
            cv2.imshow("Prediction", nn_output)

        nn_output = cv2.resize(nn_output, (origin_w, origin_h), interpolation=cv2.INTER_LANCZOS4)
        masking_threshold = 100
        masking_max = detector.max_RGB
        _, mask = cv2.threshold(nn_output,
                                masking_threshold,
                                masking_max,
                                cv2.THRESH_BINARY)

        mask = mask.astype(np.uint8)

        out_img = Image.fromarray(mask)
        output_path = os.path.join(out_dir, filename)
        out_img.save(output_path)

        if _SHOW:
            cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    processPhotos(_INPUT_DIR, _OUT_DIR)
