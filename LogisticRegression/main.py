import pandas as pd
from lr import LogisticRegression
from Scorer.scorer import BinaryScorer
from matplotlib import pyplot as plt

# Use pandas to read the whole csv
train = pd.read_csv('spambasetrain.csv', header=None)
test = pd.read_csv('spambasetest.csv', header=None)

# Prepare training sets
x_tr, y_tr = (train.iloc[:, :-1]).as_matrix(), (train.iloc[:, -1]).as_matrix().transpose()

# Prepare test sets
x_ts, y_ts = (test.iloc[:, :-1]).as_matrix(), (test.iloc[:, -1]).as_matrix().transpose()

# Instantiate the Logistic Regression model
model = LogisticRegression(l2penalty=0, verbose=True)


# This is a callback that is called for each iteration while fitting
def accuracy_callback(_model):
    # For each iteration, predict the train & test values
    pred_tr = _model.predict(x_tr)
    pred_ts = _model.predict(x_ts)

    # Evaluate
    train_scores = BinaryScorer(y_tr, pred_tr, description='Training', positive_class=1,
                                negative_class=0)
    test_scores = BinaryScorer(y_ts, pred_ts, description='Testing', positive_class=1,
                               negative_class=0)

    accuracies_tr.append(train_scores.accuracy())
    accuracies_ts.append(test_scores.accuracy())


# Evaluate and plot the model with different learning rates
for exp in (-2, -4, -6):

    accuracies_tr = []
    accuracies_ts = []
    iterations = [i for i in range(1, 1001)]

    # Fit the model. Note that while fitting we are predicting due to the presence of a callback.
    model.fit(x_tr, y_tr, lr=10**exp, maxit=1000, callback=accuracy_callback)

    # Plot the accuracies
    fig = plt.figure()

    plt.plot(iterations, accuracies_tr, '-', label='Train', color='blue')
    plt.plot(iterations, accuracies_ts, '-', label='Test', color='green')
    plt.title('Train vs Test with lr = 10^%s' % exp)
    plt.ylabel('Accuracy')
    plt.xlabel('Iteration')
    plt.grid(True, linestyle=':')
    plt.legend(loc="upper right")
    plt.show()

    fig.savefig('logistic_accuracy_lr10^%s.png' % exp, dpi=300)