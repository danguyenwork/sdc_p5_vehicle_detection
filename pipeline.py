from moviepy.editor import VideoFileClip
from etl import load_data
from extractor import Extractor
from searcher import Searcher
from frame import Frame
from trainer import Trainer
import os
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import sys
import glob
import cv2
import time
from tracker import Tracker

def pipeline(img, fname = None):
    # frame = Frame(img)
    frame = Frame(img)

    t1 = time.time()

    # Perform search here
    car_windows = searcher.search(img)
    t2 = time.time()
    print('Search time: ', t2-t1)
    frame.save_car_windows(car_windows)
    tracker.append_frame(frame)

    final_result, imgd = tracker.track(write_images)
    t3 = time.time()
    print('Track time:', t3-t2)

    if (write_images and imgmode) or (write_images and not imgmode and tracker.frame_count % 10 == 0):
        frame.append_to_image_dict(imgd)
        frame.save_plot(fname=fname)

    return final_result

def train_model():
    model_fn = 'data/finalized_model.sav'
    extractor_fn = 'data/finalized_extractor.sav'

    if not os.path.isfile(model_fn) or not os.path.isfile(extractor_fn) or OVERWRITE_TRAINER:
        print('Training')

        data = load_data()

        extractor = Extractor()

        trainer = Trainer(data, extractor)
        trainer.train()
        model = trainer.best_model

        # save the model to disk
        pickle.dump(model, open(model_fn, 'wb'))
        pickle.dump(extractor, open(extractor_fn, 'wb'))

    model = pickle.load(open(model_fn, 'rb'))
    extractor = pickle.load(open(extractor_fn, 'rb'))

    return model, extractor

if __name__ == '__main__':
    OVERWRITE_TRAINER = False

    model, extractor = train_model()
    searcher = Searcher(model, extractor)
    tracker = Tracker()

    start = int(sys.argv[1])
    end = int(sys.argv[2])
    write_images = (sys.argv[3] == 'True')
    imgmode = (sys.argv[4] == 'img')


    if imgmode:
        test_fnames = glob.glob('test_images/*.jpg')
        for fname in test_fnames:
            img = mpimg.imread(fname)
            pipeline(img, fname = fname)
            tracker = Tracker()
    else:
        test_videos = glob.glob('test_videos/test_video.mp4')
        for fname in test_videos:
            if end != 0:
                inp = VideoFileClip(fname).subclip(start,end)
            else:
                inp = VideoFileClip(fname)
            output = inp.fl_image(pipeline)
            output_dir = fname.split(".")[0] + '_output_' + str(start) + '_' + str(end) + '_.mp4'
            output.write_videofile(output_dir, audio=False)
