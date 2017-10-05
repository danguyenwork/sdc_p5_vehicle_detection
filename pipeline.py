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

def pipeline(img, fname = None):
    # frame = Frame(img)


    # Perform transformation here


    # Perform search here
    frame = Frame(img)
    draw_img = searcher.search(frame, save_intermediate_step = write_images, fname = fname)
    # if fname:
        # mpimg.imsave('output_images/' + fname.split("/")[1], draw_img)

def train_model():
    trainer_fn = 'data/finalized_trainer.sav'
    extractor_fn = 'data/finalized_extractor.sav'

    if not os.path.isfile(trainer_fn) or not os.path.isfile(extractor_fn):
        print('Training')

        data = load_data()

        model_choice = 'LinearSVC'
        params = {}
        trainer = Trainer(model_choice, params)

        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = data

        extractor = Extractor()

        x_train = extractor.fit_transform(train_features[:10])
        y_train = train_labels[:10]

        trainer.fit(x_train, y_train)

        # save the model to disk
        pickle.dump(trainer, open(trainer_fn, 'wb'))
        pickle.dump(extractor, open(extractor_fn, 'wb'))

    trainer = pickle.load(open(trainer_fn, 'rb'))
    extractor = pickle.load(open(extractor_fn, 'rb'))

    return trainer.model, extractor

if __name__ == '__main__':
    model, extractor = train_model()
    searcher = Searcher(model, extractor)

    start = int(sys.argv[1])
    end = int(sys.argv[2])
    write_images = (sys.argv[3] == 'True')
    imgmode = (sys.argv[4] == 'img')

    if imgmode:
        test_fnames = glob.glob('test_images/*.jpg')
        for fname in test_fnames:
            img = mpimg.imread(fname)
            pipeline(img, fname = fname)
            break
    else:
        test_videos = glob.glob('test_videos/*.mp4')
        for fname in test_videos:
            if end != 0:
                inp = VideoFileClip(fname).subclip(start,end)
            else:
                inp = VideoFileClip(fname)
            output = inp.fl_image(pipeline)
            output_dir = fname.split(".")[0] + '_output_' + str(start) + '_' + str(end) + '_.mp4'
            output.write_videofile(output_dir, audio=False)
