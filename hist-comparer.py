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

def extract_different_frames(vid, output_dir, max_correlation=0.82, frames_to_process=5000):
    cam = cv2.VideoCapture(vid)
    hist_len = 10
    # Overall, left and right bottom part hists
    prev_hist = []
    lb_prev_hist = []
    rb_prev_hist = []
    saved_frames = 0
    for fr in range(0, frames_to_process):
        ret_val, original = cam.read()
        if not ret_val:
            print("No video frame captured: video at end or no video present.")
            break
        (height, width, _) = original.shape
        h_middle, v_middle = int(width/2), int(height/2)
        # grey = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist(images=[# original,
                                         original[v_middle:, :h_middle],
                                         original[v_middle:, h_middle:]],
                            channels=[0,1,2],
                            mask=None,
                            histSize=[hist_len, hist_len, hist_len],
                            ranges=[0, 256, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        if fr == 0:
            # Edge case for the first frame.
            prev_hist = hist


        corr = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
        if corr < max_correlation:
            prev_hist = np.array(hist)
            print("     Different frame!!!")
            saved = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            saved.save(os.path.join(output_dir, "saved_frame_%03d.jpg" % saved_frames))
            saved_frames += 1
        print("Frame  %d corr %f" % (fr, corr))
        cv2.imshow("Frame", original)
        if cv2.waitKey(1) == 27:
            break  # esc to quit

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
    #move_night_photos_from_folder(_INPUT_DIR, _OUTPUT_DIR)
    extract_different_frames(vid='video/road8.mp4',
                             output_dir='data/from_vid',
                             max_correlation=0.7)

