import numpy as np
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
import cv2
import matplotlib.pyplot as plt

class Extractor(object):
    # BIN SPATIAL
    SPATIAL_SIZE = (32, 32)
    COLOR_SPACE = 'YCrCb'

    # COLOR HISTOGRAM
    NBINS = 32
    BINS_RANGE = (0, 256)

    # HOG
    ORIENT = 9
    PIX_PER_CELL = (8, 8)
    CELL_PER_BLOCK = (2,2)

    def scale(self, X):
        return self.scaler.transform(X)

    def set_scaler(self, X):
        self.scaler = StandardScaler().fit(X)

    def convert_color(self, x, color_space):
        color_mapping = {
            'HSV': cv2.COLOR_RGB2HSV,
            'LUV': cv2.COLOR_RGB2LUV,
            'HLS': cv2.COLOR_RGB2HLS,
            'YUV': cv2.COLOR_RGB2YUV,
            'YCrCb': cv2.COLOR_RGB2YCrCb
        }

        # Convert image to new color space (if specified)
        if color_space != 'RGB':
            x_converted = cv2.cvtColor(x, color_mapping[color_space])
        else:
            x_converted = x
        return x_converted

    # Pass the color_space flag as 3-letter all caps string
    # like 'HSV' or 'LUV' etc.
    def extract_bin_spatial(self, x, spatial_size):
        # Use cv2.resize().ravel() to create the feature vector
        x_feature = cv2.resize(x, spatial_size).ravel()

        # Return the feature vector
        return x_feature

    def extract_histogram(self, x, nbins, bins_range):
        channel_1_hist = np.histogram(x[:,:,0], bins=nbins, range=bins_range)
        channel_2_hist = np.histogram(x[:,:,1], bins=nbins, range=bins_range)
        channel_3_hist = np.histogram(x[:,:,2], bins=nbins, range=bins_range)

        hist_features = np.concatenate((channel_1_hist[0], channel_2_hist[0], channel_3_hist[0]))
        return hist_features

    def _extract_hog_one_channel(self, x, visualise = False, feature_vec = True):
        result = hog(x, orientations = self.ORIENT, pixels_per_cell = self.PIX_PER_CELL, cells_per_block = self.CELL_PER_BLOCK, visualise = visualise, feature_vector = feature_vec, transform_sqrt=False, block_norm ='L1')

        if visualise:
            hog_feature, hog_image = result
        else:
            hog_feature = result
            hog_image = None
        return hog_feature, hog_image

    def _extract_hog(self, x_converted, visualise = False, feature_vec = True, channel = None):
        if channel:
            x_feature_hog, x_image_hog = self._extract_hog_one_channel(x_converted[:,:,channel], visualise = visualise, feature_vec = feature_vec)
        else:
            x_feature_hog = []
            x_image_hog = []
            for channel in range(x_converted.shape[2]):
                feature_hog, image_hog = self._extract_hog_one_channel(x_converted[:,:,channel], visualise = visualise, feature_vec = feature_vec)
                x_feature_hog.append(feature_hog)
                x_image_hog.append(image_hog)

        if visualise:
            return np.array(x_feature_hog), x_image_hog
        else:
            return np.array(x_feature_hog), None

    def transformation_pipeline_one(self, x):
        '''
        Perform the series of transformation on a single image
        Input: image in numpy array format
        Output: un-normalized arrays of feature vectors
        '''

        x_converted = self.convert_color(x, self.COLOR_SPACE)

        x_feature_bin = self.extract_bin_spatial(x_converted, self.SPATIAL_SIZE)

        x_feature_hist = self.extract_histogram(x_converted, self.NBINS, self.BINS_RANGE)

        x_feature_hog, _ = self._extract_hog(x_converted)
        x_feature_hog = np.concatenate(x_feature_hog)

        features = np.concatenate(
            (x_feature_bin
            , x_feature_hist
            , x_feature_hog
        ))

        return features


    def fit_transform(self, X):
        features = []

        for x in X:
            features.append(self.transformation_pipeline_one(x))
        self.set_scaler(features)

        return self.scale(features)

    def transform(self, X):
        features = []

        for x in X:
            features.append(self.transformation_pipeline_one(x))
        return self.scale(features)
