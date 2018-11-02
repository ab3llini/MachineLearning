class BinaryScorer:
    def __init__(self, real, pred, positive_class='spam', negative_class='ham', description=None):

        self.tp, self.tn, self.fp, self.fn = None, None, None, None

        self.real = real
        self.pred = pred
        self.description = description

        self.positive_class = positive_class
        self.negative_class = negative_class

    def confusion_matrix(self):

        self.tp, self.tn, self.fp, self.fn = 0, 0, 0, 0

        for i, r in enumerate(self.real):

            pred_i = self.pred[i]

            if r == self.positive_class and pred_i == self.positive_class:
                self.tp += 1
            if r == self.negative_class and pred_i == self.negative_class:
                self.tn += 1
            if r == self.negative_class and pred_i == self.positive_class:
                self.fp += 1
            if r == self.positive_class and pred_i == self.negative_class:
                self.fn += 1

        return self.tp, self.tn, self.fp, self.fn

    def accuracy(self):
        if self.tp is None:
            self.confusion_matrix()

        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def precision(self):

        if self.tp is None:
            self.confusion_matrix()

        return self.tp / (self.tp + self.fp)

    def recall(self):
        if self.tp is None:
            self.confusion_matrix()

        return self.tp / (self.tp + self.fn)

    def f_score(self):
        if self.tp is None:
            self.confusion_matrix()

        return 2 * self.precision() * self.recall() / (self.precision() + self.recall())

    def describe(self):

        tp, tn, fp, fn = self.confusion_matrix()

        if self.description is not None:
            print('Description of %s scores' % self.description)

        print('Confusion matrix:')
        print('TP = %s' % tp)
        print('TN = %s' % tn)
        print('FP = %s' % fp)
        print('FN = %s\n' % fn)

        print('Measures:')
        print('ACCURACY = %s' % self.accuracy())
        print('PRECISION = %s' % self.precision())
        print('RECALL = %s' % self.recall())
        print('F-SCORE = %s\n\n' % self.f_score())
