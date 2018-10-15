import pandas as pd
import random
import numpy as np
import math


# Parser and formatter for iris dataset
class IrisParser:

    def __init__(self, df='iris.csv'):
        self.df = pd.read_csv(df, header=None)

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

        return x, np.array(y).transpose().tolist()

    def k_fold(self, x, y, k):
        size = len(y)
        fold_size = math.floor(size / k)
        rest = size % k
        x_folds = []
        y_folds = []
        for i in range(k):
            idx = i * fold_size
            if i == k - 1:
                x_folds.append(x[idx:idx + fold_size + rest])
                y_folds.append(y[idx:idx + fold_size + rest])

            else:
                x_folds.append(x[idx:idx + fold_size])
                y_folds.append(y[idx:idx + fold_size])

        return x_folds, y_folds


def _flip_val(v, seed=None):
    if seed:
        random.seed(seed)
    return random.choice([2, 3]) if v == 1 else random.choice([1, 3]) if v == 2 else random.choice([1, 2])


# Helper to generate 4 "dirty" dataset
def flip_df(seed=None):
    df = pd.read_csv('iris.csv', header=None)
    m = [10, 20, 30, 50]
    if seed:
        random.seed(seed)
    for v in m:
        # Possible flippable indices
        candidates = [i for i in range(0, df.shape[0])]
        selected = []
        # Randomly select and index in the candidates
        # and removes it from the candidates list
        for pick in range(v):
            selected_idx = random.choice(candidates)
            selected.append(selected_idx)
            candidates.remove(selected_idx)

        dirty = df.copy()
        for idx in selected:
            print('old = %s' % dirty.iloc[idx, 2])
            dirty.iloc[idx, 2] = _flip_val(dirty.iloc[idx, 2], seed)
            print('new = %s' % dirty.iloc[idx, 2])
        dirty.to_csv('iris_m'+str(v), header=False, index=False)






