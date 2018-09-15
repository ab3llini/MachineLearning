import codecs
from model import MultinomialNB
from preprocessing import Preprocessor
from scorer import BinaryScorer
from matplotlib import pyplot as plt

# Open file and read content in a variable.
# Couldn't use standard python way of opening files due to ASCII decode errors.
raw = codecs.open('SMSSpamCollection.txt', 'r', encoding='utf-8').readlines()
model = MultinomialNB()
x_tr, y_tr, x_ts, y_ts = Preprocessor(data=raw).preprocess().tokenize().split(percentage_train=0.8, functional=False)

duration = model.fit(x_tr, y_tr)

print('Fit complete')

# pred_tr = model.predict(x_tr, alpha=0.1, voc_size=20000)
# pred_ts = model.predict(x_ts, alpha=0.1, voc_size=20000)

# train_scores = BinaryScorer(y_tr, pred_tr, description='Training').describe()

i = [i for i in range(-5, 1)]
alphas = [2**exp for exp in i]
points = []

for a in alphas:

    pred_tr = model.predict(x_tr, alpha=a, voc_size=20000)
    pred_ts = model.predict(x_ts, alpha=a, voc_size=20000)

    print('predicted values for %s' % a)

    scores_tr = BinaryScorer(y_tr, pred_tr)
    scores_ts = BinaryScorer(y_ts, pred_ts)

    points.append([
        scores_tr.accuracy(),
        scores_ts.accuracy(),
        scores_tr.f_score(),
        scores_ts.f_score()
    ])

plt.subplot(2, 1, 1)
plt.plot(i, points[:, 0], 'o-')
plt.plot(i, points[:, 1], 'o-')
plt.title('Train vs Test')
plt.ylabel('Precision measure')

plt.subplot(2, 1, 2)
plt.plot(i, points[:, 2], i, points[:, 3], '.-')
plt.xlabel('i value')
plt.ylabel('F-Score measure')

plt.show()


