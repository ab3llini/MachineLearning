import codecs
import numpy as np
from model import MultinomialNB
from preprocessing import Preprocessor
from scorer import BinaryScorer
from matplotlib import pyplot as plt

# Open file and read content in a variable.
# Couldn't use standard python way of opening files due to ASCII decode errors.
raw = codecs.open('SMSSpamCollection.txt', 'r', encoding='utf-8').readlines()

# Create a Multinomial Naive Bayes Classifier, in this case we only have 2 classes so it is Binary
model = MultinomialNB()

# Preprocess, Tokenize and Split data in train and test
# IMPORTANT: Unless seed parameter is removed from call, the split will always be the same.
x_tr, y_tr, x_ts, y_ts = Preprocessor(data=raw).preprocess().tokenize().split(percentage_train=0.8, seed=5555, functional=False)

# Fit the model
model.fit(x_tr, y_tr)

print('Fit complete')

# Predict train and test values
pred_tr = model.predict(x_tr, alpha=0.1, voc_size=20000)
pred_ts = model.predict(x_ts, alpha=0.1, voc_size=20000)

# Print train and test predictions performance
train_scores = BinaryScorer(y_tr, pred_tr, description='Training').describe()
test_scores = BinaryScorer(y_ts, pred_ts, description='Training').describe()

# Create list of alphas with different i value
i = [i for i in range(-5, 1)]
alphas = [2**exp for exp in i]
points = []

# For each alpha, compute predictions for train & test, check their score and add points to be plotted later
for a in alphas:

    pred_tr = model.predict(x_tr, alpha=a, voc_size=20000)
    pred_ts = model.predict(x_ts, alpha=a, voc_size=20000)

    print('Done predicting values with alpha=%s' % a)

    scores_tr = BinaryScorer(y_tr, pred_tr)
    scores_ts = BinaryScorer(y_ts, pred_ts)

    points.append([
        scores_tr.accuracy(),
        scores_ts.accuracy(),
        scores_tr.f_score(),
        scores_ts.f_score()
    ])

# Convert points to np array to use axis selector
points = np.array(points)

# Creates one single plot with two subplots
# The upper one shows train vs test results for accuracy
# The lower one shows train vs test results for f score
fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(i, points[:, 0], 'o-', label='Train', color='blue')
plt.plot(i, points[:, 1], 'o-', label='Test', color='orange')
plt.title('Train vs Test')
plt.ylabel('Accuracy')
plt.grid(True, linestyle=':')
plt.legend(loc="upper right")

plt.subplot(2, 1, 2)
plt.plot(i, points[:, 2], 'o-', label='Train', color='green')
plt.plot(i, points[:, 3], 'o-', label='Test', color='purple')
plt.xlabel('i value')
plt.ylabel('F-Score')
plt.grid(True, linestyle=':')
plt.legend(loc="upper right")

plt.show()
fig.savefig('graph.png', dpi=300)


