import codecs
from NaiveBayes.model import MultinomialNB
from NaiveBayes.preprocessing import Preprocessor
from NaiveBayes.scorer import BinaryScorer

# Open file and read content in a variable.
# Couldn't use standard python way of opening files due to ASCII decode errors.
raw = codecs.open('SMSSpamCollection.txt', 'r', encoding='utf-8').readlines()
model = MultinomialNB(verbose=True)
x_tr, y_tr, x_ts, y_ts = Preprocessor(data=raw).preprocess().tokenize().split(percentage_train=0.8, functional=False)

model.fit(x_tr, y_tr)

pred_tr = model.predict(x_tr, alpha=0.1, voc_size=20000)
pred_ts = model.predict(x_ts, alpha=0.1, voc_size=20000)

train_scores = BinaryScorer(y_tr, pred_tr, description='Training').describe()
test_scores = BinaryScorer(y_ts, pred_ts, description='Testing').describe()

