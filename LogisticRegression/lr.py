from model import Model
import numpy as np
import math
from tqdm import tqdm


class LogisticRegression(Model):

    """
    This class performs logistic regression using gradient descent
    Upon prior fit, it is able to predict the class for a specific input
    """

    def __init__(self, l2penalty=0.0, verbose=False):
        """
        Initializes the instance
        :param l2penalty: value for Ridge Regularization, defaults to 0
        :param verbose: if on, enables debugging mode
        """

        self.l2penalty = l2penalty
        self.verbose = verbose
        self.weights = None

    def __logistic_for(self, x):
        return 1 / (1 + math.e ** - float(np.dot(self.weights, x)))

    def __exp_for(self, x):
        return math.e ** np.dot(self.weights, x)

    def __compute_gradient(self, x, y):

        """
        Computes the gradient for the iteration that calls this method
        The computation is based on the current weights
        :param x: the input matrix of features
        :param y: the input column of labels
        :return: the gradient
        """

        gradient = np.zeros(shape=(1, x.shape[1]))

        for x_i, y_i in zip(x, y):
            # Compute the logistic function value for x just once for performance reasons
            logistic = self.__exp_for(x_i)

            # Compute the contribution of this sample to the gradient for the considered weight
            gradient += x_i * y_i - (x_i * logistic) / (1 + logistic) - self.l2penalty * self.weights

        return gradient

    def fit(self, x, y, lr=10**(-4), maxit=1000, tolerance=math.e**(-2), callback=None):

        """
        This method fits the model
        :param x: input vector of features
        :param y: input vector of labels
        :param lr: learning rate
        :param maxit: maximum number of iterations
        :param tolerance: threshold to stop fitting
        :return: self, allows to make functional calls
        :param callback: if provided will be called passing the status of the model at each iteration
        """

        # Useful variables
        feature_size = x.shape[1]

        # Convert input to np arrays
        x = x if type(x) is not np.ndarray else np.array(x)
        y = y if type(y) is not np.ndarray else np.array(y)

        # Initialize weights
        self.weights = np.zeros(shape=(1, feature_size))

        # Perform gradient ascent
        for it in tqdm(range(maxit)) if not self.verbose else range(maxit):

            # Initialize gradient to a zero vector
            gradient = self.__compute_gradient(x, y)

            # Update weights
            self.weights += lr * gradient

            # mean abs gradient value
            mag = np.mean(np.abs(gradient))

            if self.verbose:
                print('It = %s, mean abs gradient = %s' % (it, mag))

            # Stop fitting if mag get below tolerance threshold
            if mag < tolerance:
                break

            # If present, perform a callback for any kind of purpose
            if callback:
                callback(self)

        return self

    def predict(self, x):

        """
        Predicts the output label for the given input
        :param x: the input vector
        :return: the predicted label
        """

        size = x.shape[0]
        out = np.empty(shape=(size, 1))

        for i, sample in enumerate(x):
            out[i] = np.round(self.__logistic_for(sample))

        return out
