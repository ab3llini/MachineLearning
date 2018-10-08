from model import Model
import distance as dist
import numpy as np
import operator


# Class that implements KNN
class KNeighborsClassifier(Model):

    # Constructor: receives number of neighbors to work with
    def __init__(self, n_neighbors=1):
        self.data = None
        self.labels = None
        self.n_neighbors = n_neighbors

    # Just saves the data and labels
    def fit(self, x, y):

        self.data = x
        self.labels = y

    # For each sample in x vector, compute the K closest neighbors and performs majority voting amongst them
    def predict(self, x):

        preds = []

        for sample in x:

            # Contains the closest neighbors
            neighbors = self.neighbors(sample)

            # Add to the prediction vector the resulting label
            preds.append(self._majority_vote(neighbors))

        return preds

    # Returns the n indexes of the closest points
    def neighbors(self, x):

        distances = []

        for point in self.data:
            # Compute the distances
            distances.append(dist.euclidean(x, point))

        # Find the n closest neighbors
        distances = np.array(distances)

        # Gives the indexes of the closest elements
        return distances.argsort()[:self.n_neighbors]

    # Performs majority voting among a set of neighbors
    def _majority_vote(self, neighbors):
        scores = {}

        # Count how many elements belong to which class
        for neighbor in neighbors:

            # We use [0] to access the element label since we have a col vector
            # and self.labels[neighbor] is an array of just one element referring
            # to some row
            _class = self.labels[neighbor]

            # Increment the class counts properly while examining the neighbors
            if _class not in scores:
                scores[_class] = 1
            else:
                scores[_class] += 1

        # Pick the class with highest number and return it
        return max(scores.items(), key=operator.itemgetter(1))[0]
