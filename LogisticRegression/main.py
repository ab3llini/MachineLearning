import pandas as pd
from lr import LogisticRegression
from Scorer.scorer import BinaryScorer
from matplotlib import pyplot as plt

# Number of iterations
ITERATIONS = 200

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

        iterations = [i for i in range(1, ITERATIONS + 1)]

        # Fit the model. Note that while fitting we are predicting due to the presence of a callback.
        model.fit(x_tr, y_tr, lr=10**exp, maxit=ITERATIONS, callback=[accuracy_callback, accuracies_tr, accuracies_ts])

        # Plot the accuracies
        fig = plt.figure()

        plt.plot(iterations, accuracies_tr.copy(), label='Train', color='blue', linewidth=1)
        plt.plot(iterations, accuracies_ts.copy(), label='Test', color='green', linewidth=1)
        plt.title('Train vs Test with lr = 10^%s' % exp)
        plt.ylabel('Accuracy')
        plt.xlabel('Iteration')
        plt.grid(True, linestyle=':')
        plt.legend(loc="upper right")
        plt.show()

        fig.savefig('logistic_accuracy_lr10^%s.png' % exp)


def compare_penalty():

    penalties = (-8, -6, -4, -2, 0, 2)

    # Instantiate the Logistic Regression model with no regularization
    model_unreg = LogisticRegression(verbose=True)

    # Fit the model.
    model_unreg.fit(x_tr, y_tr, lr=10 ** (-4), maxit=ITERATIONS, tolerance=None)

    # Predict results
    pred_tr_unreg = model_unreg.predict(x_tr)
    pred_ts_unreg = model_unreg.predict(x_ts)

    # Evaluate
    train_scores = BinaryScorer(y_tr, pred_tr_unreg, description='Training', positive_class=1,
                                negative_class=0)
    test_scores = BinaryScorer(y_ts, pred_ts_unreg, description='Testing', positive_class=1,
                               negative_class=0)

    accuracy_tr_unreg = train_scores.accuracy()
    accuracy_ts_unreg = test_scores.accuracy()

    accuracies_tr_reg = []
    accuracies_ts_reg = []

    # Evaluate and plot the model with different learning rates
    for alpha in penalties:

        # Instantiate the Logistic Regression model with regularization
        model_reg = LogisticRegression(l2penalty=2**alpha, verbose=True)

        # Fit the model.
        model_reg.fit(x_tr, y_tr, lr=10**(-4), maxit=ITERATIONS, tolerance=None)

        # Predict results
        pred_tr_reg = model_reg.predict(x_tr)
        pred_ts_reg = model_reg.predict(x_ts)

        # Evaluate
        train_scores = BinaryScorer(y_tr, pred_tr_reg, description='Training', positive_class=1,
                                    negative_class=0)
        test_scores = BinaryScorer(y_ts, pred_ts_reg, description='Testing', positive_class=1,
                                   negative_class=0)

        accuracies_tr_reg.append(train_scores.accuracy())
        accuracies_ts_reg.append(test_scores.accuracy())

    fig = plt.figure()

    plt.plot(penalties, accuracies_tr_reg, 'o-', label='Train w/ reg', color='blue')
    plt.plot(penalties, accuracies_ts_reg, 'o-', label='Test w/ reg', color='purple')
    plt.plot([penalties[0], penalties[len(penalties) - 1]], [accuracy_tr_unreg, accuracy_tr_unreg], '-', label='Train w/o reg', color='red')
    plt.plot([penalties[0], penalties[len(penalties) - 1]], [accuracy_ts_unreg, accuracy_ts_unreg], '-', label='Test w/o reg', color='orange')

    plt.title('Regularization comparison, lr = 10^-4')
    plt.ylabel('Accuracy')
    plt.xlabel('Values of k, with alpha = 2^k')
    plt.grid(True, linestyle=':')
    plt.legend(loc="upper right")

    plt.show()
    fig.savefig('logistic_penalty_comp.png', dpi=300)


compare_lr()
compare_penalty()