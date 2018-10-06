from model import Model
import distance as dist
import numpy as np
import operator


class KNeighborsClassifier(Model):

    def __init__(self, n_neighbors=1):
        self.data = None
        self.labels = None
        self.n_neighbors = n_neighbors

    def fit(self, x, y):

        self.data = x
        self.labels = y

    def predict(self, x):

        preds = []

        for sample in x:

            neighbors = self.neighbors(sample)

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

    def _majority_vote(self, neighbors):
        scores = {}

        # Count how many elements belong to which class
        for neighbor in neighbors:
            _class = self.labels[neighbor][0]

            if _class not in scores:
                scores[_class] = 0
            else:
                scores[_class] += 1

        # Pick the class with highest number

        return max(scores.items(), key=operator.itemgetter(1))[0]
