import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from extractor import Extractor
from scipy.ndimage.measurements import label

class Searcher(object):
    def __init__(self, model, extractor, x_start_stop = (0, 1280), y_start_stop = (0,720), xy_window = (64,64), xy_overlap = (0.5, 0.5), width = 1280, height = 720):
        self.model = model

        hyperparams = self._calculate_search_hyperparameters(x_start_stop, y_start_stop, xy_window, xy_overlap,width, height)

        self.x_start = hyperparams[0]
        self.x_stop = hyperparams[1]
        self.y_start = hyperparams[2]
        self.y_stop = hyperparams[3]
        self.sliding_pixel_x = hyperparams[4]
        self.sliding_pixel_y = hyperparams[5]

        self.width = width
        self.height = height

        self.xy_window = xy_window

        self.extractor = extractor

    def _calculate_search_hyperparameters(self, x_start_stop, y_start_stop, xy_window, xy_overlap, width, height):
        x_start = x_start_stop[0] or 0
        x_stop = x_start_stop[1] or width
        y_start = y_start_stop[0] or 0
        y_stop = min(y_start_stop[1], height // 2) if y_start_stop[1] else height // 2

        sliding_pixel_x = int(xy_window[0] * xy_overlap[0])
        sliding_pixel_y = int(xy_window[1] * xy_overlap[1])

        return x_start, x_stop, y_start, y_stop, sliding_pixel_x, sliding_pixel_y

    # Here is your draw_boxes function from the previous exercise
    def draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            # Need to reverse the tuple because cv2.rectangle expects coordinates in the (x, y) order
            cv2.rectangle(imcopy, bbox[0][::-1], bbox[1][::-1], color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    # Define a function that takes an image,
    # start and stop positions in both x and y,
    # window size (x and y dimensions),
    # and overlap fraction (for both x and y)
    def _slide_window(self, img):
        window_list = []

        bottom_left = (self.height, 0)
        top_right = (self.height - self.xy_window[1], self.xy_window[0])

        while True:
            window_list.append((bottom_left, top_right))

            # move the bottom_left point sliding_pixel_x pixels left
            bottom_left = (bottom_left[0], bottom_left[1] + self.sliding_pixel_x)

            # move the top_right point sliding_pixel_x left
            top_right = (top_right[0], top_right[1] + self.sliding_pixel_x)

            # if the box goes over the right border, move up to the next row and start over from the left
            if top_right[1] > self.x_stop:
                bottom_left = (bottom_left[0] - self.sliding_pixel_y, 0)
                top_right = (top_right[0] - self.sliding_pixel_y, self.xy_window[1])

            # if the box exceeds the top border, the search is done
            if top_right[0] < self.y_stop:
                break

        return window_list

    def _add_heat(self, img, windows):
        heatmap = np.zeros_like(img[:,:,0]).astype(np.float)

        for window in windows:
            heatmap[window[1][0]:window[0][0], window[0][1]:window[1][1]] += 1

        return np.clip(heatmap, 0, 255)

    def _apply_threshold(self, heatmap, threshold):
        copy = np.copy(heatmap)
        copy[heatmap <= threshold] = 0
        return copy

    def _predict(self, img, window):
        y_start = window[1][0]
        y_end = window[0][0]
        x_start = window[0][1]
        x_end = window[1][1]

        test_img = cv2.resize(img[y_start:y_end, x_start:x_end], (64, 64)).astype(float)

        feature_vector = self.extractor.transform([test_img])

        prediction = self.model.predict(feature_vector)
        return prediction[0]

    def _draw_labeled_bboxes(self,img, labels):
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
        # Return the image
        return img

    def search(self, frame, save_intermediate_step = False, fname = None):
        img = frame.get_img_to_predict()

        windows = self._slide_window(img)
        car_windows = []
        for window in windows:
            prediction = self._predict(img, window)

            if prediction:
                car_windows.append(window)

        frame.windows_without_filtering = self.draw_boxes(img, car_windows)

        heatmap_no_thresh = self._add_heat(img, car_windows)
        frame.append_to_image_dict('heatmap_without_threshold', heatmap_no_thresh)

        heatmap_thresh = self._apply_threshold(heatmap_no_thresh, 1)
        frame.append_to_image_dict('heatmap_with_threshold', heatmap_thresh)

        labels = label(heatmap_thresh)
        frame.append_to_image_dict('car_labels_on_heatmap', labels[0])

        draw_img = self._draw_labeled_bboxes(np.copy(img), labels)
        frame.append_to_image_dict('car_labels_on_original', draw_img)

        if save_intermediate_step:
            frame.save_plot(fname=fname)

        return draw_img
