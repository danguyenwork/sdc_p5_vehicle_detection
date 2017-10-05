import numpy as np
from sklearn.preprocessing import StandardScaler

class Extractor(object):
    def flatten(self, x):
        self.feature_vector = np.reshape(x, (len(x), -1))
        return self.feature_vector

    def fit_scale(self, x):
        self.scaler = StandardScaler().fit(x)

    def transform_scale(self, x):
        return self.scaler.transform(x)

    def fit_transform(self, x):
        copy = self.flatten(x)
        self.fit_scale(copy)
        return self.transform_scale(copy)

    def transform(self, x):
        copy = self.flatten(x)
        return self.transform_scale(copy)
