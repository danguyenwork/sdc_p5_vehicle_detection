from collections import deque
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage.measurements import label
from collections import OrderedDict


class Tracker(object):

    N_FRAME_HEATMAP_THRESHOLD = 7

    def __init__(self, nframe = 10):
        self.nframe = nframe
        self.queue = deque()
        self.current_img = None
        self.frame_count = 0

    def append_frame(self, frame):
        if len(self.queue) == self.nframe:
            self.queue.popleft()
        self.queue.append(frame.car_windows)
        self.windows = [window for window in self.queue if window]
        if self.windows:
            self.windows = np.concatenate(self.windows)
        self.current_img = frame.img()
        self.frame_count += 1

    def track(self, write_images = False):
        img = self.current_img

        heatmap_no_thresh = self._add_heat_to_heatmap(img, self.windows)
        heatmap_thresh = self._apply_threshold_on_heatmap(heatmap_no_thresh, self.N_FRAME_HEATMAP_THRESHOLD)
        labels = label(heatmap_thresh)
        labeled_bboxes = self._find_labeled_bboxes(labels)
        final_result = self._draw_boxes(img, labeled_bboxes)

        if write_images:
            imgd = OrderedDict()

            imgd['windows_without_filtering'] = self._draw_boxes(img, self.windows)
            imgd['heatmap_without_threshold'] = heatmap_no_thresh
            imgd['heatmap_with_threshold'] = heatmap_thresh
            imgd['car_labels_on_heatmap'] = labels[0]
            imgd['car_labels_on_original'] = final_result
        else:
            imgd = None

        return final_result, imgd

    # Here is your draw_boxes function from the previous exercise
    def _draw_boxes(self, img, bboxes, color=(0, 0, 255), thick=6):
        # Make a copy of the image
        imcopy = np.copy(img)
        # Iterate through the bounding boxes
        for bbox in bboxes:
            # Draw a rectangle given bbox coordinates
            # Need to reverse the tuple because cv2.rectangle expects coordinates in the (x, y) order
            first_corner = tuple(bbox[0][::-1])
            second_corner = tuple(bbox[1][::-1])
            cv2.rectangle(imcopy, first_corner, second_corner, color, thick)
        # Return the image copy with boxes drawn
        return imcopy

    # Heatmap
    def _add_heat_to_heatmap(self, img, windows):
        heatmap = np.zeros_like(img[:,:,0]).astype(np.float)

        for window in windows:
            heatmap[window[0][0]:window[1][0], window[0][1]:window[1][1]] += 1

        return np.clip(heatmap, 0, 255)

    def _apply_threshold_on_heatmap(self, heatmap, threshold):
        copy = np.copy(heatmap)
        copy[heatmap <= threshold] = 0
        return copy

    # Merge labels into bboxes
    def _find_labeled_bboxes(self,labels):
        bboxes = []
        # Iterate through all detected cars
        for car_number in range(1, labels[1]+1):
            # Find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            # Define a bounding box based on min/max x and y
            bbox = (np.min(nonzeroy),(np.min(nonzerox))),( np.max(nonzeroy),(np.max(nonzerox)))
            # Draw the box on the image
            bboxes.append(bbox)
        # Return the image
        return bboxes
