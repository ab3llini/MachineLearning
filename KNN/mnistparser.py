from scipy.io import loadmat
import random
import sys
import math
import numpy as np


# This class parses the mnist dataset and returns x and y vectors for training
class MnistParser:

    def __init__(self, target=0):

        # Load the matlab file
        self.df = loadmat('mnist.mat')['mnist'][0][0]

        # Convert data from uint8 to int32 to avoid overflows and unexpected behaviour during computation
        self.tr_images = np.array(self.df[target]).astype(np.int32)
        self.tr_labels = np.array(self.df[target + 2]).astype(np.int32)

    # Returns an image at index idx
    def fetch_img(self, idx):
        return self.tr_images[:, :, idx], self.tr_labels[idx]

    # Select size samples from the dataset, shuffle is turned on by default
    def select(self, seed=None, shuffle=True, size=1000):
        if seed is not None:
            random.seed(seed)

        x, y = [], []

        # Size of the dataset
        ltr = len(self.tr_labels)

        # Used as a helper to shuffle the images preserving proper label
        container = []

        # Fetch all the images
        for idx in range(ltr):
            img, label = self.fetch_img(idx)
            container.append([img, label])

        # If wanted, shuffle
        if shuffle:
            random.shuffle(container)

        # Select randomly a subset of container and use it as training set
        start_idx = random.randint(0, ltr - 1)

        # Method used to select a portion of a data frame, given start and end index
        x, y = self._subsplit(container, start_idx, size)

        return x, y

    # This method creates and returns k folds for the x and y vectors
    def k_fold(self, x, y, k):

        size = len(y)
        fold_size = math.floor(size / k)

        # Depending on size and k, we might need to account even for non equal folds
        rest = size % k
        x_folds = []
        y_folds = []

        # Creates and append elements to each fold
        for i in range(k):
            idx = i * fold_size
            if i == k - 1:
                x_folds.append(x[idx:idx + fold_size + rest])
                y_folds.append(y[idx:idx + fold_size + rest])

            else:
                x_folds.append(x[idx:idx + fold_size])
                y_folds.append(y[idx:idx + fold_size])

        return x_folds, y_folds

    @staticmethod
    def _subsplit(data, start, size):

        # Containers for x and y (either train, validation or test)
        x, y = [], []

        # Starting index
        idx = start
        data_size = len(data)

        # Counter to keep track of how many samples have been added to the list
        added = 0
        while added < size:
            if idx == data_size:
                idx = 0

            # This is the current sample we are analyzing
            sample = data[idx]

            x.append(sample[0])
            y.append(sample[1][0])

            # Increment both counter and index
            added += 1
            idx += 1

        return x, y

    # Aux method used for debug to print an image on console
    def print_img_repr(self, img):
        for r in img:
            for c in r:
                if c > 0:
                    sys.stdout.write("*")
                else:
                    sys.stdout.write("-")

            sys.stdout.write("\n")


