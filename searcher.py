import matplotlib.pyplot as plt
import numpy as np
import cv2
from extractor import Extractor
import collections
import itertools
import time

class Searcher(object):
    X_START = 0
    X_STOP = 1280
    Y_START = 400
    Y_STOP = 656

    CELLS_PER_STEP = 2

    SCALES = [1, 1.4, 1.8, 2.2, 2.5]
    PREDICT_PROBA_THRESHOLD = 0.90

    def __init__(self, model, extractor):
        self.model = model
        self.extractor = extractor

    def _predict_window_one(self, img, window):
        y_start = window[1][0]
        y_end = window[0][0]
        x_start = window[0][1]
        x_end = window[1][1]

        x = cv2.resize(img[y_start:y_end, x_start:x_end], (64, 64)).astype(np.uint8)


        feature_vector = self.extractor.transform([x])

        prediction = self.model.predict(feature_vector)
        return prediction[0]

    def _search_by_hog_subsampling_one(self, img, scale):
        extractor = self.extractor
        ystart = self.Y_START
        ystop = self.Y_STOP

        orient = extractor.ORIENT
        pix_per_cell = extractor.PIX_PER_CELL[0]
        cell_per_block = extractor.CELL_PER_BLOCK[0]
        spatial_size = extractor.SPATIAL_SIZE
        hist_bins = extractor.NBINS
        bins_range = extractor.BINS_RANGE

        num_cells = pix_per_cell
        cells_per_step = self.CELLS_PER_STEP
        window = num_cells * pix_per_cell

        img_tosearch = img[ystart:ystop,:,:]
        x_converted = extractor.convert_color(img_tosearch, extractor.COLOR_SPACE)

        if scale != 1:
            imshape = x_converted.shape
            x_converted = cv2.resize(x_converted, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))

        # Define blocks and steps as above
        nxblocks = (x_converted.shape[1] // pix_per_cell) - cell_per_block + 1
        nyblocks = (x_converted.shape[0] // pix_per_cell) - cell_per_block + 1

        nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1

        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image

        # start = time.time()

        hog1, hog2, hog3 = extractor._extract_hog(x_converted, feature_vec = False)[0]

        # end = time.time()

        # print('hog: ', end - start)

        car_windows = []

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos, xpos = yb * cells_per_step, xb * cells_per_step
                xleft, ytop = xpos * pix_per_cell,  ypos * pix_per_cell

                # Extract the image patch
                subimg = cv2.resize(x_converted[ytop:ytop+window, xleft:xleft+window], (64,64))

                # Get features
                x_feature_bin = extractor.extract_bin_spatial(subimg, spatial_size)

                x_feature_hist = extractor.extract_histogram(subimg, hist_bins, bins_range)

                x_feature_hog = np.hstack(map(lambda hog_channel: hog_channel[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel(), [hog1, hog2, hog3]))

                features = np.hstack((x_feature_bin, x_feature_hist, x_feature_hog))

                # Scale features and make a prediction
                test_features = extractor.scale([features])
                predict_proba = self.model.predict_proba(test_features)[0]
                # test_prediction = self.model.predict(test_features)[0]

                # if test_prediction:
                if predict_proba[1] > self.PREDICT_PROBA_THRESHOLD:
                    # import ipdb; ipdb.set_trace()
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)

                    car_windows.append(((ytop_draw+ystart, xbox_left), (ytop_draw+win_draw+ystart, xbox_left+win_draw)))

        return car_windows

    # Main search method

    def search(self, img):
        car_windows = []

        for scale in self.SCALES:
            car_windows.extend(self._search_by_hog_subsampling_one(img, scale))

        return car_windows
