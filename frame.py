import collections
import matplotlib.pyplot as plt
import seaborn
import numpy as np
import datetime

class Frame(object):
    def __init__(self, img):
        imgd = collections.OrderedDict()
        imgd['original'] = img
        self.imgd = imgd

    def append_to_image_dict(self, name, img):
        self.imgd[name] = img

    def get_img_to_predict(self):
        return self.imgd['original']

    def save_plot(self,fname=None):
        output_dir = 'output_images/frame/'
        if not fname:
            d = datetime.utcnow()
            fname = str(unixtime) + ".jpg"
        else:
            fname = fname.split("/")[1]
        output_name = output_dir + fname

        fig, arr = plt.subplots(len(self.imgd.keys())//2+1,2,sharex=True, figsize = (12,7))
        arr = np.concatenate(arr)
        ind = 0
        for k, v in self.imgd.items():
            mode = 'gray' if len(v.shape) == 2 else 'viridis'
            arr[ind].imshow(v,cmap=mode)
            arr[ind].set_title(str(ind + 1) + '_' + k, loc='left')
            ind += 1
        plt.tight_layout()
        plt.savefig(output_name)
