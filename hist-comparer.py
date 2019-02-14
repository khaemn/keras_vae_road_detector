import numpy as np
from PIL import Image
import os
import cv2
import datetime
import shutil


_INPUT_DIR = 'data/original'
_OUTPUT_DIR = 'data/filtered'
_MAX_CORRELATION = 0.5

np.set_printoptions(suppress=True, precision=2)

def filter_video_frames(vid, output_dir, max_correlation=0.5):
    return

def move_night_photos_from_folder(input_dir, output_dir, verbose=False):
    in_files = []
    for root_back, dirs_back, files_back in os.walk(input_dir):
        for _file in files_back:
            in_files.append(_file)
    total_files = len(in_files)
    print("%s %d files found for processing" % (str(datetime.datetime.now()), total_files))
    if (total_files < 1):
        print("Not enough files for filtering. At least 2 files needed.")
        return
    current_index = 0
    hist_len = 5
    moved_files = 0

    while current_index < total_files:
        print("Processing file %d of %d" % (current_index+1, total_files))
        fname = in_files[current_index]
        in_path = os.path.join(input_dir, fname)
        curr_img = cv2.imread(in_path)
        curr_img = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)

        curr_hist = cv2.calcHist(images=[curr_img],
                                 channels=[0],
                                 mask=None,
                                 histSize=[hist_len],
                                 ranges=[0, 256])
        curr_hist = cv2.normalize(curr_hist, curr_hist).flatten()
        left = np.sum(curr_hist[:1])
        right = np.sum(curr_hist[1:])
        isDay = left < right
        if verbose:
            print(curr_hist, fname, 'DAY' if isDay else 'NIGHT')
        if not isDay:
            out_path = os.path.join(output_dir, fname)
            shutil.move(in_path, out_path)
            moved_files += 1
            if verbose:
                print("    %s moved to filtered dir" % fname)
        current_index += 1
    print("%s\nProcessed %d files, %d files moved" % (str(datetime.datetime.now()), total_files, moved_files))


if __name__ == "__main__":
    move_night_photos_from_folder(_INPUT_DIR, _OUTPUT_DIR)

