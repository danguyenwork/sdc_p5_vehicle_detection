import os
import pickle
import numpy as np
from PIL import Image
from sklearn.utils import shuffle
from tqdm import tqdm
from zipfile import ZipFile

DATA_DIRECTORY = 'data/'

def uncompress_features_labels(file):
    """
    Uncompress features and labels from a zip file
    :param file: The zip file to extract the data from
    """
    features = []
    labels = []
    sources = []

    with ZipFile(file) as zipf:
        # Progress Bar
        filenames_pbar = tqdm(zipf.namelist(), unit='files')

        # Get features and labels from all files
        for filename in filenames_pbar:
            # Check if the file is a directory
            if not filename.endswith('.png') or '__MACOSX' in filename :
                continue

            with zipf.open(filename) as image_file:
                image = Image.open(image_file)
                image.load()
                # Load image data as 1 dimensional array
                # We're using float32 to save on memory space
                feature = np.array(image, dtype=np.float32)

            # Get the the letter from the filename.  This is the letter of the image.
            label = 0 if 'non-vehicle' in filename else 1

            if 'GTI' in filename:
                source = 'GTI'
            elif 'KITTI' in filename:
                source = 'KITTI'
            else:
                source = 'Udacity'

            features.append(feature)
            labels.append(label)
            sources.append(source)
    return np.array(features), np.array(labels), np.array(sources)

def extract_data_from_zip(docker_size_limit=150000):
    # sample size
    # GTI vehicle - 2826
    # KTTI vehicle - 5966
    # Non vehicle - 8968

    # the training data consists of:
    # - all the GTI vehicle data (2826 pictures)
    # - 55% of the KITTI vehicle data (3281 pictures)
    # - 70% of the non-vehicles data (6277 pictures)

    # the validation data consists of:
    # - 30% of the KITTI vehicle data (1790 pictures)
    # - 20% of the non-vehicle data (1793 pictures)

    # the test data consists of:
    # - 15% of the KITTI vehicle data (895 pictures)
    # - 10% of the non-vehicle data (897 pictures)

    # Extract vehicle and non vehicle data from zip files
    vehicle_features, vehicle_labels, vehicle_sources = uncompress_features_labels('data/vehicles.zip')
    non_vehicle_features, non_vehicle_labels, non_vehicle_sources = uncompress_features_labels('data/non-vehicles.zip')

    # Shuffle the data
    vehicle_features, vehicle_labels, vehicle_sources = shuffle(vehicle_features, vehicle_labels, vehicle_sources)
    non_vehicle_features, non_vehicle_labels, non_vehicle_sources = shuffle(non_vehicle_features, non_vehicle_labels, non_vehicle_sources)

    # Create masks for the three sources: GTI, KITTI and Udacity
    # Split features and labels by sources based on those masks

    GTI_mask = (vehicle_sources == 'GTI')
    KITTI_mask = (vehicle_sources == 'KITTI')

    GTI_features, GTI_labels = vehicle_features[GTI_mask], vehicle_labels[GTI_mask]
    KITTI_features, KITTI_labels = vehicle_features[KITTI_mask], vehicle_labels[KITTI_mask]

    # Split data by percentage so we can combine them into train, valid and test sets
    # Example: We are splitting KITTI by 55 - 30 - 15 so we would pass .55 and .85
    # to get_index and get back [3281, 5071]
    # then we can split by [:3281], [3281:5071], [5071:]

    get_index = lambda arr, ind: (arr.shape[0] * np.array(ind)).astype(int)

    KITTI_ind = get_index(KITTI_features, [.55, .85])
    non_vehicle_ind = get_index(non_vehicle_features, [.70, .90])

    KITTI_features_split = np.split(KITTI_features, KITTI_ind)
    KITTI_labels_split = np.split(KITTI_labels, KITTI_ind)

    non_vehicle_features_split = np.split(non_vehicle_features, non_vehicle_ind)
    non_vehicle_labels_split = np.split(non_vehicle_labels, non_vehicle_ind)

    # combine the subarrays into train, valid and test sets

    train_features = np.concatenate((GTI_features, KITTI_features_split[0], non_vehicle_features_split[0]))
    valid_features = np.concatenate((KITTI_features_split[1], non_vehicle_features_split[1]))
    test_features = np.concatenate((KITTI_features_split[2], non_vehicle_features_split[2]))

    train_labels = np.concatenate((GTI_labels, KITTI_labels_split[0], non_vehicle_labels_split[0]))
    valid_labels = np.concatenate((KITTI_labels_split[1], non_vehicle_labels_split[1]))
    test_labels = np.concatenate((KITTI_labels_split[2], non_vehicle_labels_split[2]))

    train_features, train_labels = shuffle(train_features, train_labels)
    valid_features, valid_labels = shuffle(valid_features, valid_labels)
    test_features, test_labels = shuffle(test_features, test_labels)

    return train_features, train_labels, valid_features, valid_labels, test_features, test_labels

def prep_data_for_training():
    train_features, train_labels, valid_features, valid_labels, test_features, test_labels = extract_data_from_zip()

    pickle_file = 'data.pickle'

    if not os.path.isfile(DATA_DIRECTORY + pickle_file):
        print('Saving data to pickle file...')
        try:
            with open('data/data.pickle', 'wb') as pfile:
                pickle.dump(
                    {
                        'train_dataset': train_features,
                        'train_labels': train_labels,
                        'valid_dataset': valid_features,
                        'valid_labels': valid_labels,
                        'test_dataset': test_features,
                        'test_labels': test_labels,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise

    print('Data cached in pickle file.')

def load_data():
    pickle_file = 'data.pickle'
    if not os.path.isfile(DATA_DIRECTORY + pickle_file):
        prep_data_for_training()
    with open(DATA_DIRECTORY + pickle_file, 'rb') as f:
      pickle_data = pickle.load(f)
      train_features = pickle_data['train_dataset']
      train_labels = pickle_data['train_labels']
      valid_features = pickle_data['valid_dataset']
      valid_labels = pickle_data['valid_labels']
      test_features = pickle_data['test_dataset']
      test_labels = pickle_data['test_labels']
      del pickle_data  # Free up memory
      return train_features, train_labels, valid_features, valid_labels, test_features, test_labels
