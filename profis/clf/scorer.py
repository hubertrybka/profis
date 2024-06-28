import pickle

import numpy as np


class SKLearnScorer:
    """
    Scorer class for Bayesian optimization, based on scikit-learn models
    Parameters:
        path: path to the model file (pickle)
    """

    def __init__(self, path):
        with open(path, "rb") as file:
            self.model = pickle.load(file)

    def __call__(self, **args) -> float:
        input_vector = list({**args}.values())
        input_vector = np.array(input_vector).reshape(1, -1)
        output = self.model.predict_proba(input_vector)[0][1]
        return output
