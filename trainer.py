from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

class Trainer(object):

    verbose = True

    def __init__(self, data, extractor):
        self.best_model = None
        self.extractor = extractor
        self.data = data

    def _prep_data(self, data):
        train_features, train_labels, valid_features, valid_labels, test_features, test_labels = data

        X_train = self.extractor.fit_transform(train_features)
        y_train = train_labels
        X_valid = self.extractor.fit_transform(valid_features)
        y_valid = valid_labels
        X_test = self.extractor.fit_transform(test_features)
        y_test = test_labels

        return X_train, y_train, X_valid, y_valid, X_test, y_test

    def _train_svm(self, data):
        X_train, y_train, X_valid, y_valid, X_test, y_test = data

        model = LinearSVC()
        calibrated_svc = CalibratedClassifierCV(model, cv=3)

        calibrated_svc.fit(X_train, y_train)
        return calibrated_svc

    def train(self):
        if self.verbose:
            print('Extracting features')

        prep_data = self._prep_data(self.data)
        X_train, y_train, X_valid, y_valid, X_test, y_test = prep_data

        if self.verbose:
            print('Training SVM')

        svm_model = self._train_svm(prep_data)

        svm_train_score = svm_model.score(X_train, y_train)
        svm_valid_score = svm_model.score(X_valid, y_valid)

        if self.verbose:
            print('Train score: ', svm_train_score)
            print('Valid score: ', svm_valid_score)

        self.best_model = svm_model

        return self.best_model
