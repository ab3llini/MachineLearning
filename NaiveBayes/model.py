import operator


class Model:
    def fit(self, x, y):
        pass

    def predict(self, y):
        pass


# This is a multi class implementation of the naive bayes classifier.
# Even though the dataset contains just two classes, it has been implemented to work with an arbitrary number of classes
class MultinomialNB(Model):

    def __init__(self, verbose=False):
        self.counts = None
        self.priors = None
        self.verbose = verbose

    # Train the model
    # Requires a matrix X of features and a column vector of labels
    def fit(self, x, y):

        # Clear counts & priors
        self.priors = {}
        self.counts = {}

        self._compute_priors(y)
        self._compute_counts(x, y)

    # Predicts output classes given an input, usually test set
    # For the specific purpose of the assignment, an additional value might be passes: voc_size
    # In this case this value is set to 20.000 by the caller.
    def predict(self, x, alpha=1.0, voc_size=None):

        # List of predicted values
        predictions = []

        # For each sample provided predict the class taking the highest score out of
        # every possible class
        for sample in x:
            # Compute the probabilities for each possible class and store it in a dict
            # Scores = { class_1 : probability, ..., class_n : probability }
            scores = {}
            strings = {}
            for class_ in self.priors:

                # Init the probability of word to the one of the prior P(class)
                p = self.priors[class_]

                # Compute total number of words for current class
                c_d = 0

                for word in self.counts:
                    c_d += self.counts[word][class_]

                strings[class_] = str(self.priors[class_])
                for token in sample:
                    # For each word compute its probability, given the class, alpha and voc_size and multiply with p
                    p *= self._compute_probability(token, class_, c_d, alpha, voc_size)

                # Add the class with its computed probability to the scores dictionary
                scores[class_] = p

            # Compute the actual prediction for this sample, taking the class with higher probability
            predicted = max(scores.items(), key=operator.itemgetter(1))[0]
            predictions.append(predicted)

        return predictions

    # Computes the probability of a word given the class
    def _compute_probability(self, w, class_, c_d, alpha, voc_size=None):

        # Check whether or not the word w appeared at least once while fitting the model or not
        null_frequency = False if w in self.counts else True

        # Number of occurrences of word xi in spam/ham messages y
        c_d_y = self.counts[w][class_] if not null_frequency else 0

        # Compute probability : p = (count_d_y * alpha) / (count_d * alpha * N)
        return (c_d_y + alpha) / (c_d + alpha * (len(self.counts) if voc_size is None else voc_size))

    # Compute prior probabilities for every class
    def _compute_priors(self, y):
        if self.verbose:
            print('Computing priors..')

        # Count size of y, will divide the class count to compute ith P(class)
        tot = len(y)
        for class_ in y:
            self.priors[class_] = 1 if class_ not in self.priors else (self.priors[class_] + 1)

        self.priors = {k: v / tot for k, v in self.priors.items()}

        if self.verbose:
            print('Priors : %s' % self.priors)
            print('Computing word counts.. this might take a couple of seconds..')

    # Creates a dictionary where the key is a word and the value is another dictionary where
    # we have two keys: ham and spam. For each of these two keys there is the number of occurrences of the word in that
    # class
    # Complexity of this step is {O(N_total_words)}, since a single scan is required over the whole data
    # Access to dictionary takes O(1) since they are basically hash tables
    def _compute_counts(self, x, y):
        for idx, sample in enumerate(x):
            for feature in sample:
                if feature in self.counts:
                    self.counts[feature][y[idx]] += 1
                else:
                    self.counts[feature] = {}
                    for class_ in y:
                        self.counts[feature][class_] = 0 if y[idx] != class_ else 1

