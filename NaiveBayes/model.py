import operator


class Model:
    def fit(self, x, y):
        pass

    def predict(self, y):
        pass


class MultinomialNB(Model):

    def __init__(self, verbose=False):
        self.counts = None
        self.priors = None
        self.verbose = verbose

    # Train the model
    # Requires a matrix X of features and a column vector of labels
    def fit(self, x, y):

        self.counts = {}
        self.priors = {}

        # Compute prior probabilities, aka P(class)
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

        self._compute_counts(x, y)

    # Predicts output classes given an inout, usually test set
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
                p = 1

                # Number of words for label spam/ham words y
                c_d = 0
                for word in self.counts:
                    c_d += self.counts[word][class_]

                strings[class_] = str(self.priors[class_])
                for token in sample:
                    # For each word compute its probability, given the class, alpha and voc_size
                    p *= self._compute_probability(token, class_, c_d, alpha, voc_size)

                    # After we computed the probabilities for each token we need to multiply by the prior P(class)
                p *= self.priors[class_]
                # Add the class with its computed probability to the scores dictionary
                scores[class_] = p

            # Compute the actual prediction for this sample, taking the class with higher probability
            predicted = max(scores.items(), key=operator.itemgetter(1))[0]
            predictions.append(predicted)

        return predictions

    # Computes the probability of a word given the class
    def _compute_probability(self, w, class_, c_d, alpha, voc_size=None):

        null_frequency = False if w in self.counts else True

        # Number of occurrences of word xi in spam/ham messages y
        c_d_y = self.counts[w][class_] if not null_frequency else 0

        # Compute probability : p = (count_d_y * alpha) / (count_d * alpha * N)
        return (c_d_y + alpha) / (c_d + alpha * (len(self.counts) if voc_size is None else voc_size))

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

