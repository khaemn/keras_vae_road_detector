from keras.models import Sequential, load_model
import numpy as np
import cv2
import time
import os
import datetime

_MODEL_FILENAME = 'models/model_vae_roader.h5'

_STACK_PREDICTIONS = False
_STACK_DEPTH = 10
_STACK_DECAY = 0.5

_FRAME_DIVIDER = 3
_TOTAL_FRAMES = 2000

class RoadDetector:
    model = Sequential()
    max_RGB = 255
    cv2.CAP_PROP_FRAME_COUNT
    input_height = 180  # 90
    input_width = 320  # 160
    # N thresholds will produce N masks of N colors
    mask_thresholds = [50, 200, 240]
    fill_colors = [[255, 50, 255], [255, 255, 50], [50, 255, 255]]

    def __init__(self, modelFile=_MODEL_FILENAME):
        self.model = load_model(modelFile)

    def predict(self, _input):
        (original_height, original_width) = _input.shape
        # Resizing to acceptable size
        if original_height != self.input_height or original_width != self.input_width:
            resized = cv2.resize(_input, (self.input_width, self.input_height))
        else:
            resized = _input
        # Gamma preprocessing to increase sensitivity
        # normalized = cv2.equalizeHist(resized)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        normalized = clahe.apply(resized)
        # Normalization
        normalized = normalized / self.max_RGB
        # Prediction
        model_input = np.array([normalized])
        model_input = model_input.reshape((1, self.input_height, self.input_width, 1))
        # Only one frame is used for prediction
        prediction = self.model.predict(model_input)
        prediction = prediction[0] * self.max_RGB

        prediction = prediction.astype('uint8')

        return prediction

    def getMaskFor(self, image):
        (origin_h, origin_w) = image.shape
        prediction = self.predict(image)

        prediction = cv2.resize(prediction, (origin_w, origin_h), interpolation=cv2.INTER_LANCZOS4)
        masking_threshold = self.mask_thresholds[0]
        masking_max = self.max_RGB
        _, mask = cv2.threshold(prediction,
                                masking_threshold,
                                masking_max,
                                cv2.THRESH_BINARY)

        mask = mask.astype(np.uint8)
        return mask

    # https://www.pyimagesearch.com/2015/10/05/opencv-gamma-correction/
    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

def simple_test():
    detector = RoadDetector()
    roadimg = cv2.imread('dataset/test/lanes1_r.jpg')
    roadimg = cv2.cvtColor(roadimg, cv2.COLOR_BGR2RGB)

    roadmask = detector.predict(roadimg)

    cv2.imshow('Origin', roadimg)
    cv2.imshow('Prediction', roadmask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_video(paths):
    # Instal codecs using    $ sudo apt-get install ubuntu-restricted-extras
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    wr_width = 1024  # 640
    wr_height = 576  # 360
    # out = cv2.VideoWriter('output.avi', -1, 20.0, (wr_width, wr_height))
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (wr_width, wr_height))
    detector = RoadDetector()

    # increase speed by dividing frames
    divider = _FRAME_DIVIDER
    frames_to_process = _TOTAL_FRAMES
    framestack = list()
    big_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    small_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    masking_threshold = detector.mask_thresholds[0]
    masking_max = RoadDetector.max_RGB

    for path in paths:
        cam = cv2.VideoCapture(path)
        for fr in range(0, frames_to_process):
            for d in range(0, divider):
                ret_val, original = cam.read()
            if not ret_val:
                print("No video frame captured: video at end or no video present.")
                break

            original = cv2.resize(original, (wr_width, wr_height))

            # flipping for some interesting results
            # original = cv2.flip(original, 0)

            dataForNN = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            dataForNN = cv2.resize(dataForNN, (RoadDetector.input_width, RoadDetector.input_height))

            # start = datetime.datetime.now()  # time.process_time()

            prediction = detector.predict(dataForNN)

            rawmask = prediction.copy()
            rawmask_size = 0.2
            rawmask = cv2.resize(rawmask, (int(wr_width * rawmask_size), int(wr_height * rawmask_size)))
            rawmask = cv2.cvtColor(rawmask, cv2.COLOR_GRAY2BGR)

            _, min_mask = cv2.threshold(prediction,
                                    masking_threshold,
                                    masking_max,
                                    cv2.THRESH_BINARY)
            # Preprocess to reduce noise
            min_mask = cv2.dilate(cv2.erode(min_mask, big_kernel, iterations=2), small_kernel, iterations=2)
            # min_mask = cv2.erode(cv2.dilate(min_mask, kernel, iterations=2), kernel, iterations=2)

            vis_min_mask = cv2.resize(min_mask, (int(wr_width * rawmask_size), int(wr_height * rawmask_size)))
            vis_min_mask = cv2.cvtColor(vis_min_mask, cv2.COLOR_GRAY2BGR)

            input_mini = cv2.resize(dataForNN, (int(wr_width * rawmask_size), int(wr_height * rawmask_size)))
            input_mini = cv2.cvtColor(input_mini, cv2.COLOR_GRAY2BGR)

            processed = cv2.resize(min_mask, (wr_width, wr_height), interpolation=cv2.INTER_LANCZOS4)
            _, mask = cv2.threshold(processed,
                                    masking_threshold,
                                    masking_max,
                                    cv2.THRESH_BINARY)
            mask = cv2.dilate(cv2.erode(mask, small_kernel, iterations=2), big_kernel, iterations=2)
            mask = mask.astype(np.uint8)

            alpha = 0.3
            combined = np.array(original, dtype=np.uint8)
            color_fill = np.array(original, dtype=np.uint8)
            color_fill[:, :] = detector.fill_colors[0]
            color_fill = cv2.bitwise_and(color_fill, color_fill, mask=mask)
            cv2.addWeighted(combined, 1 - alpha, color_fill, alpha, 0, combined)

            # overlaying a raw mask image to top left corner
            y_vis_offs = rawmask.shape[0]
            x_vis_offs = rawmask.shape[1]
            combined[:y_vis_offs, :x_vis_offs] = cv2.resize(input_mini, (x_vis_offs, y_vis_offs))
            combined[:y_vis_offs,
                     x_vis_offs:x_vis_offs*2] = vis_min_mask
            combined[:y_vis_offs,
                     x_vis_offs*2:x_vis_offs*3] = vis_min_mask

            # Printing the threshold value
            text = "Threshold %02d of %02d" % (masking_threshold, masking_max)
            cv2.putText(combined, text, (int(wr_width / 3), rawmask.shape[0] + 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(combined, text, (int(wr_width / 3) + 2, rawmask.shape[0] + 30 + 2),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(combined,
                        "%s %d of %d" % (os.path.basename(path), fr, frames_to_process),
                        (int(wr_width / 3), 30),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(combined,
                        "%s %d of %d" % (os.path.basename(path), fr, frames_to_process),
                        (int(wr_width / 3) + 2, 30 + 1),
                        cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

            # do some stuff
            # end = datetime.datetime.now()  # time.process_time()
            # elapsed = end - start
            # print("elapsed time %d" % int(elapsed.total_seconds() * 1000))

            cv2.imshow('Prediction', combined)

            out.write(combined)

            # cv2.imshow('Origin', cv2.cvtColor(dataForNN, cv2.COLOR_RGB2BGR))
            # cv2.imshow('Roadmask', prediction)

            if cv2.waitKey(1) == 27:
                break  # esc to quit
    cam.release()
    out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    process_video([
                    # 'video/road10.mp4',
                    # 'video/road9.mp4',
                    'video/road16.mp4',
                    'video/road15.mp4',
                    # 'video/road11.mp4',
                    # 'video/road12.mp4',
                    # 'video/road13.mp4',
                    # 'video/road14.mp4',
                    # 'video/road1.mp4',
                    # 'video/noroad_1.mp4',
                    # 'video/road2.mp4',
                    # 'video/noroad_2.mp4',
                    # 'video/road3.mp4',
                    # 'video/noroad_3.mp4',
                    # 'video/road4.mp4',
                    # 'video/noroad_4.mp4',
                    # 'video/road5.mp4',
                    # 'video/noroad_5.mp4',
                    'video/road6.mp4',
                    # 'video/noroad_6.mp4',
                    # 'video/road7.mp4',
                    # 'video/noroad_7.mp4',
                    # 'video/road8.mp4',
                    # 'video/diy-road7.3gp',
                    # 'video/diy-road8.3gp',
                    # 'video/diy-road11.3gp',
                    # 'video/diy-road12.3gp',
                    # 'video/test/test-road-1.mp4',
                    # 'video/test/test-road-2.mp4',
                    # 'video/test/test-road-3.mp4',
                    'video/kiev/kiev2.mp4',
                    'video/kiev/kiev5.mp4',
    ])