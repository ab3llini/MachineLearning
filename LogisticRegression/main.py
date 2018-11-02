import pandas as pd
from lr import LogisticRegression

# Use pandas to read the whole csv
train = pd.read_csv('spambasetrain.csv', header=None)
test = pd.read_csv('spambasetest.csv', header=None)

# Prepare training sets
x_tr, y_tr = (train.iloc[:, :-1]).as_matrix(), (train.iloc[:, -1]).as_matrix().transpose()

# Prepare test sets
x_ts, y_ts = (test.iloc[:, :-1]).as_matrix(), (test.iloc[:, -1]).as_matrix().transpose()

# Instantiate the Logistic Regression model
model = LogisticRegression(l2penalty=0, verbose=True)
# Fit the model
model.fit(x_tr, y_tr, lr=10**-4, maxit=1000)

