import numpy as np
from PIL import Image
import os
from sys import maxsize
import cv2
import datetime
import shutil


_INPUT_DIR = 'data/original'
_OUTPUT_DIR = 'data/filtered'
_MAX_CORRELATION = 0.5

_OUT_HEIGHT = 640
_OUT_WIDTH = 360

np.set_printoptions(suppress=True, precision=2)


def extract_different_frames(vid,
                             output_dir,
                             max_correlation=0.82,
                             frames_to_process=maxsize,
                             at_least=20):
    cam = cv2.VideoCapture(vid)
    hist_len = 8
    # Overall, left and right bottom part hists
    prev_l_hist = []
    prev_r_hist = []
    saved_frames = 0
    prev_fr = 0
    vid_length = int(cam.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = int(vid_length / at_least)  # at least 20 frames from a video, event if they are not different
    frames_to_process = min(frames_to_process, vid_length)
    for fr in range(0, frames_to_process):
        ret_val, original = cam.read()
        if not ret_val:
            print("No video frame captured: video at end or no video present.")
            break
        (height, width, _) = original.shape
        h_middle, v_middle = int(width/2), int(height/2)
        grey = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        l_hist = cv2.calcHist(images=[grey[:, :h_middle]],
                              channels=[0],
                              mask=None,
                              histSize=[hist_len],
                              ranges=[0, 256])
        r_hist = cv2.calcHist(images=[grey[:, h_middle:]],
                              channels=[0],
                              mask=None,
                              histSize=[hist_len],
                              ranges=[0, 256])
        l_hist = cv2.normalize(l_hist, l_hist).flatten()
        r_hist = cv2.normalize(r_hist, r_hist).flatten()
        if fr == 0:
            # Edge case for the first frame.
            prev_r_hist = r_hist
            prev_l_hist = l_hist

        r_corr = cv2.compareHist(prev_r_hist, r_hist, cv2.HISTCMP_CORREL)
        l_corr = cv2.compareHist(prev_l_hist, l_hist, cv2.HISTCMP_CORREL)
        pause_ms = 1
        print("Frame  %d l_corr %f r_corr %f" % (fr, l_corr, r_corr))
        if l_corr < max_correlation \
                or r_corr < max_correlation \
                or fr == 0 \
                or fr - prev_fr > interval:
            prev_fr = fr
            prev_r_hist = np.array(r_hist)
            prev_l_hist = np.array(l_hist)
            print("     Different frame saved.")
            savename = os.path.splitext(os.path.basename(vid))[0]
            saved = Image.fromarray(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
            saved.save(os.path.join(output_dir,
                                    "%s_%04d.jpg" % (savename,
                                                     saved_frames)))
            cv2.rectangle(original,
                          (0, v_middle),
                          (width, height),
                          (0, 0, 255),
                          -1)
            saved_frames += 1
            pause_ms = 50
        cv2.putText(original,
                    "%s %d of %d" % (os.path.basename(vid), fr, frames_to_process),
                    (int(width / 3), 30),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(original,
                    "%s %d of %d" % (os.path.basename(vid), fr, frames_to_process),
                    (int(width / 3) + 2, 30 + 1),
                    cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow("Processing ...", original)

        if cv2.waitKey(pause_ms) == 27:
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

    paths = [
        # ['O:/Datasets/vae_roader_custom/video/diy-road7.3gp', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/diy-road8.3gp', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/diy-road11.3gp', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/diy-road12.3gp', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/noroad_1.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/noroad_2.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/noroad_3.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/noroad_4.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/noroad_5.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/noroad_6.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/noroad_7.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/road1.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/road2.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/road3.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/road4.mp4', 0.72],
        # ['O:/Datasets/vae_roader_custom/video/road5.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/road6.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/road7.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/road8.mp4', 0.7],
        # ['O:/Datasets/vae_roader_custom/video/road9.mp4', 0.65],
        # ['O:/Datasets/vae_roader_custom/video/road10.mp4', 0.65],
        # ['O:/Datasets/vae_roader_custom/video/road11.mp4', 0.65],
        # ['O:/Datasets/vae_roader_custom/video/road12.mp4', 0.65],
        # ['O:/Datasets/vae_roader_custom/video/road13.mp4', 0.65],
        # ['O:/Datasets/vae_roader_custom/video/road14.mp4', 0.65],
        # ['O:/Datasets/vae_roader_custom/video/road15.mp4', 0.65],
        ['/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/video/kiev/kiev1.mp4', 0.65],
        ['/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/video/kiev/kiev2.mp4', 0.65],
        ['/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/video/kiev/kiev3.mp4', 0.65],
        ['/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/video/kiev/kiev4.mp4', 0.65],

    ]

    for v in paths:
        extract_different_frames(vid=v[0],
                                 # output_dir='data/from_vid',
                                 output_dir='/media/rattus/40F00470F0046F0A/Datasets/vae_roader_custom/fkievvid',
                                 # fkievvid
                                 max_correlation=v[1],
                                 at_least=30)

