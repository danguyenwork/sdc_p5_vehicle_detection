from sklearn.svm import LinearSVC

class Trainer(object):

    algmap = {'LinearSVC': LinearSVC}

    def __init__(self, model_choice, params):
        self.model_choice = model_choice
        params['verbose'] = True
        self.model = self.algmap[self.model_choice](**params)
        self.scaler = None


    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    # def predict(x):
    #
    #
    # def score(x, y_true):
