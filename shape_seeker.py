# from keras.models import Sequential, load_model
import numpy as np
from PIL import Image
import os
import cv2
from video_processor import RoadDetector
from scipy.interpolate import splprep, splev
import datetime

# https://www.pyimagesearch.com/2016/02/08/opencv-shape-detection/
# https://agniva.me/scipy/2016/10/25/contour-smoothing.html
# https://stackoverflow.com/questions/41879315/opencv-using-cv2-approxpolydp-correctly
# https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/

_MODEL_FILENAME = 'models/micro_model_yolike_roader.h5'
_INPUT_DIR = '/home/rattus/Projects/PythonNN/datasets/downloaded-assorted/masks/'
_OUT_DIR = '/home/rattus/Projects/PythonNN/datasets/downloaded-assorted/genmasks/'
_SHOW = False

# TODO: move into a class
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

def smoothContour(cont):
    x, y = cont.T
    # Convert from numpy arrays to normal arrays
    x = x.tolist()[0]
    y = y.tolist()[0]
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splprep.html
    tck, u = splprep([x, y], u=None, s=1.0, per=1)
    # https://docs.scipy.org/doc/numpy-1.10.1/reference/generated/numpy.linspace.html
    u_new = np.linspace(u.min(), u.max(), 35)  # 25
    # https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.interpolate.splev.html
    x_new, y_new = splev(u_new, tck, der=0)
    # Convert it back to numpy format for opencv to be able to display it
    res_array = [[[int(i[0]), int(i[1])]] for i in zip(x_new, y_new)]
    return np.asarray(res_array, dtype=np.int32)

def getRoadOutline(binary_img):
    canny_thres1 = 100
    canny_thres2 = 200
    approx_epsilon = 0.005
    cropping_offset = 2  # pixels
    (h_, w_) = binary_img.shape

    # Preprocess to reduce noise
    mask = cv2.erode(cv2.dilate(binary_img.copy(), kernel, iterations=2), kernel)
    mask = cv2.dilate(cv2.erode(mask, kernel, iterations=2), kernel)

    # Cutting 1-pixel line from all sides of the image to ensure Canny
    # will get a proper contour
    canned = mask
    canned[:cropping_offset, :] = 0
    canned[:, :cropping_offset] = 0
    canned[h_-cropping_offset:, :] = 0
    canned[:, w_-cropping_offset:] = 0
    canned = cv2.Canny(canned, canny_thres1, canny_thres2)

    if (_SHOW):
        cv2.imshow("Canned", canned)
    i, cnts, h = cv2.findContours(canned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    approxed = []
    for contour in cnts:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, approx_epsilon * peri, True)
        approxed.append(approx)

    if len(approxed) > 1:
        # In case of multiple contours, we take the largest one
        areas = []
        for contour in approxed:
            areas.append(cv2.contourArea(contour))
        max_index = areas.index(max(areas))
        approxed = [approxed[max_index]]
    return approxed

def getVerticalMiddleSplittingLine(mask):
    # Outline is an array of points, encircling road outline on the image
    # This function tries to define bottom left and right points of
    # triangle-alike-shaped road outline, then find the topmost point,
    # assuming it is near horizon AND near visible road ending in infinity.
    # Then vertical length of the road is being splitted to N parts and for each
    # of these heights two points on the outline are found.
    # Ignore above text, this method does different thing.

    total_h_steps = 20
    (height, width) = mask.shape

    lefts, centers, rights = [], [], []
    top_margin = 0.25
    top_offset = int(total_h_steps * top_margin)
    height_step, bottom_offset = divmod(int(height * (1 - top_margin)), total_h_steps)
    for step in range(0, total_h_steps):
        y = int(height * top_margin) + step * height_step
        prev_pix = mask[y, 0]

        # Find left border (0 to 255) or right border (255 to 0) at the plane
        left_ind, right_ind, center_ind = -1, -1, -1
        for index in range(1, width, 5):
            pix = mask[y, index]
            if pix > prev_pix or (pix == 255 and index == 1):
                if left_ind < 0:
                    lefts.append([index, y])
                    left_ind = index
            if pix < prev_pix or (pix == 255 and index == width - 1):
                if right_ind < 0 and left_ind > 0:
                    rights.append([index, y])
                    right_ind = index
            if left_ind > 0 and right_ind > 0:
                if center_ind < 0:
                    center_ind = int(left_ind + (right_ind - left_ind) / 2)
                    centers.append([center_ind, y])
                pass
            prev_pix = pix
    # print("Found %d lefts, %d rights, %d centers" % (len(lefts), len(rights), len(centers)))
    lefts, centers, rights = np.array(lefts, dtype='int32'), np.array(centers, dtype='int32'), np.array(rights, dtype='int32')

    if _SHOW:
        drawn = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cv2.polylines(drawn, [lefts], False, (255, 255, 0), 2)
        cv2.polylines(drawn, [centers], False, (0, 255, 255), 2)
        cv2.polylines(drawn, [rights], False, (255, 0, 255), 2)
        cv2.imshow("Lined", drawn)
    return lefts, centers, rights

def processPhotos(input_dir=_INPUT_DIR, out_dir=_OUT_DIR):
    # detector = RoadDetector(_MODEL_FILENAME)
    files = []

    for root_back, dirs_back, files_back in os.walk(input_dir):
        for _file in files_back:
            files.append(_file)

    total_files = len(files)
    iteration = 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for filename in files:
        print('Processing file', iteration, 'of', total_files)
        iteration += 1

        x_img = cv2.imread(os.path.join(_INPUT_DIR, filename))

        nn_output = cv2.cvtColor(x_img, cv2.COLOR_BGR2GRAY)
        masking_threshold = 100
        masking_max = RoadDetector.max_RGB
        _, mask = cv2.threshold(nn_output,
                                masking_threshold,
                                masking_max,
                                cv2.THRESH_BINARY)

        if _SHOW:
            cv2.imshow("Raw", mask)
        mask = cv2.dilate(mask, kernel)
        mask = cv2.erode(mask, kernel)
        if _SHOW:
            cv2.imshow("Refined", mask)

        getVerticalMiddleSplittingLine(mask)

        outline = getRoadOutline(mask)

        cv2.drawContours(x_img, outline, -1, (255, 0, 255), 2)
        if _SHOW:
            cv2.imshow("Outline", x_img)

        # out_img = Image.fromarray(mask)
        # output_path = os.path.join(out_dir, filename)
        # out_img.save(output_path)

        if _SHOW:
            cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_videos(paths):
    detector = RoadDetector()
    wr_width, wr_height = 640, 360
    frames_to_process = 2000
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('outline-output.avi', fourcc, 20.0, (wr_width, wr_height))

    for path in paths:
        cam = cv2.VideoCapture(path)
        for fr in range(0, frames_to_process):
            ret_val, original = cam.read()
            if not ret_val:
                print("No video frame captured: video at end or no video present.")
                break

            original = cv2.resize(original, (wr_width, wr_height))

            # flipping for some interesting results
            # original = cv2.flip(original, 0)

            dataForNN = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

            start = datetime.datetime.now()  # time.process_time()

            prediction = detector.predict(dataForNN)

            rawmask = prediction.copy()
            # cv2.imshow('Rawmask', rawmask)
            rawmask_size = 0.25
            rawmask = cv2.resize(rawmask, (int(wr_width * rawmask_size), int(wr_height * rawmask_size)))
            rawmask = cv2.cvtColor(rawmask, cv2.COLOR_GRAY2BGR)

            # prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2GRAY)
            processed = cv2.resize(prediction, (wr_width, wr_height), interpolation=cv2.INTER_LANCZOS4)
            masking_threshold = detector.mask_thresholds[0]
            masking_max = detector.max_RGB
            _, mask = cv2.threshold(processed,
                                    masking_threshold,
                                    masking_max,
                                    cv2.THRESH_BINARY)

            outline = getRoadOutline(mask)
            outline_poly = np.zeros_like(mask, dtype='uint8')
            cv2.drawContours(outline_poly, outline, -1, (255), 2)
            if len(outline) > 3:
                cv2.fillPoly(outline_poly, outline, 255, 2)
                cv2.imshow("OPOly", outline_poly)
                cv2.waitKey(0)
            # outline = np.zeros(mask)

            lefts, centers, rights = getVerticalMiddleSplittingLine(mask)
            # drawn = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            cv2.polylines(original, [lefts], False, (255, 255, 0), 2)
            cv2.polylines(original, [centers], False, (0, 255, 255), 2)
            cv2.polylines(original, [rights], False, (255, 0, 255), 2)

            # overlaying a raw mask image to top left corner
            original[:rawmask.shape[0], :rawmask.shape[1]] = rawmask

            end = datetime.datetime.now()  # time.process_time()
            elapsed = end - start
            print("elapsed time %d" % int(elapsed.total_seconds() * 1000))

            cv2.imshow('Prediction', original)
            out.write(original)

            if cv2.waitKey(1) == 27:
                break  # esc to quit
    out.release()
    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # processPhotos(_INPUT_DIR, _OUT_DIR)
    # quit()
    process_videos([
        'video/road9.mp4',
        'video/road10.mp4',
        'video/road11.mp4',
        'video/road12.mp4',
        'video/road13.mp4',
        'video/road14.mp4',
        'video/road15.mp4',
        'video/road1.mp4',
        'video/noroad_1.mp4',
        'video/road2.mp4',
        'video/noroad_2.mp4',
        'video/road3.mp4',
        'video/noroad_3.mp4',
        'video/road4.mp4',
        'video/noroad_4.mp4',
        'video/road5.mp4',
        'video/noroad_5.mp4',
        'video/road6.mp4',
        'video/noroad_6.mp4',
        'video/road7.mp4',
        'video/noroad_7.mp4',
        'video/road8.mp4',
        'video/diy-road7.3gp',
        'video/diy-road8.3gp',
        'video/diy-road11.3gp',
        'video/diy-road12.3gp',
    ])