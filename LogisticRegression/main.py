import pandas as pd
from lr import LogisticRegression
from Scorer.scorer import BinaryScorer
from matplotlib import pyplot as plt

# Use pandas to read the whole csv
train = pd.read_csv('spambasetrain.csv', header=None)
test = pd.read_csv('spambasetest.csv', header=None)

# Add a column of ones to account for w0 (intercept)
train.insert(loc=0, column='const', value=1)
test.insert(loc=0, column='const', value=1)

# Prepare training sets
x_tr, y_tr = (train.iloc[:, :-1]).values, (train.iloc[:, -1]).values.transpose()


# Prepare test sets
x_ts, y_ts = (test.iloc[:, :-1]).values, (test.iloc[:, -1]).values.transpose()


# This is a callback that is called for each iteration while fitting
def accuracy_callback(*args):

    _model, acc_tr, acc_ts = args

    # For each iteration, predict the train & test values
    pred_tr = _model.predict(x_tr)
    pred_ts = _model.predict(x_ts)

    # Evaluate
    train_scores = BinaryScorer(y_tr, pred_tr, description='Training', positive_class=1,
                                negative_class=0)
    test_scores = BinaryScorer(y_ts, pred_ts, description='Testing', positive_class=1,
                               negative_class=0)

    acc_tr.append(train_scores.accuracy())
    acc_ts.append(test_scores.accuracy())


def compare_lr():
    # Instantiate the Logistic Regression model
    model = LogisticRegression(l2penalty=0, verbose=True)

    # Evaluate and plot the model with different learning rates
    for exp in (-2, -4, -6):

        accuracies_tr = []
        accuracies_ts = []

        iterations = [i for i in range(1, 1001)]

        # Fit the model. Note that while fitting we are predicting due to the presence of a callback.
        model.fit(x_tr, y_tr, lr=10**exp, maxit=1000, callback=[accuracy_callback, accuracies_tr, accuracies_ts])

        # Plot the accuracies
        fig = plt.figure()

        plt.plot(iterations, accuracies_tr.copy(), label='Train', color='blue', linewidth=1)
        plt.plot(iterations, accuracies_ts.copy(), label='Test', color='green', linewidth=1)
        plt.title('Train vs Test with lr = 10^-%s' % exp)
        plt.ylabel('Accuracy')
        plt.xlabel('Iteration')
        plt.grid(True, linestyle=':')
        plt.legend(loc="upper right")
        plt.show()

        fig.savefig('logistic_accuracy_lr10^-%s.svg' % exp)


def compare_penalty():

    # Evaluate and plot the model with different learning rates
    for alpha in (-8, -6, -4, -2, 0, 2):

        accuracies_tr = []
        accuracies_ts = []

        # Instantiate the Logistic Regression model
        model = LogisticRegression(l2penalty=2**alpha, verbose=True)

        iterations = [i for i in range(1, 1001)]

        # Fit the model. Note that while fitting we are predicting due to the presence of a callback.
        model.fit(x_tr, y_tr, lr=10**(-4), maxit=1000, tolerance=None, callback=[accuracy_callback, accuracies_tr, accuracies_ts])

        # Plot the accuracies
        fig = plt.figure()

        plt.plot(iterations, accuracies_tr.copy(), label='Train', color='blue', linewidth=1)
        plt.plot(iterations, accuracies_ts.copy(), label='Test', color='green', linewidth=1)
        plt.title('Train vs Test with lr = 10^-4 and alpha = 2^%s' % alpha)
        plt.ylabel('Accuracy')
        plt.xlabel('Iteration')
        plt.grid(True, linestyle=':')
        plt.legend(loc="upper right")
        plt.show()

        fig.savefig('logistic_accuracy_lr10^-4_alpha2^%s.svg' % alpha)


compare_penalty()