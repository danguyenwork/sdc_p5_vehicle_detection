import collections
import matplotlib.pyplot as plt
import seaborn
import numpy as np
from datetime import datetime
from skimage.feature import hog

class Frame(object):
    def __init__(self, img):
        imgd = collections.OrderedDict()
        imgd['original'] = img
        self.imgd = imgd
        self.car_windows = None

    def img(self):
        return self.imgd['original']

    def save_car_windows(self, car_windows):
        self.car_windows = car_windows

    def append_to_image_dict(self, imgd):
        self.imgd.update(imgd)

    def get_img_to_predict(self):
        return self.imgd['original']

    def save_plot(self,fname=None):
        output_dir = 'output_images/frame/'
        if not fname:
            unixtime = datetime.utcnow()
            fname = str(unixtime) + ".jpg"
        else:
            fname = fname.split("/")[1]
        output_name = output_dir + fname

        fig, arr = plt.subplots( (len(self.imgd.keys()) + 1) // 2, 2, sharex=True, figsize = (12,8))
        arr = np.concatenate(arr.reshape(-1,1))
        ind = 0
        for k, v in self.imgd.items():
            mode = 'gray' if len(v.shape) == 2 else 'viridis'
            arr[ind].imshow(v,cmap=mode)
            arr[ind].set_title(str(ind + 1) + '_' + k, loc='left')
            ind += 1
        plt.tight_layout()
        plt.savefig(output_name)
        plt.clf()
