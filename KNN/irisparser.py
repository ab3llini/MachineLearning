import pandas as pd
import random
import numpy as np


# Parser and formatter for iris dataset
class IrisParser:

    def __init__(self):
        self.df = pd.read_csv('iris.csv', header=None)

    # Returns a sample from the dataset as (features, label)
    def fetch_sample(self, idx):
        return self.df.iloc[idx, :2], self.df.iloc[idx, -1]

    # Parses the dataset into arrays of features and labels
    def parse(self, seed=None, shuffle=True):
        if seed is not None:
            random.seed(seed)

        x, y = [], []

        ltr = self.df.shape[0]

        # Used as a helper to shuffle the images preserving proper label
        container = []

        for idx in range(ltr):
            features, label = self.fetch_sample(idx)
            x.append(features)
            y.append(label)

        if shuffle:
            random.shuffle(container)

        return np.array(x), np.array(y).transpose()




