from mnistparser import MnistParser
from KNN import KNeighborsClassifier
from tqdm import tqdm
import numpy as np
import plotter as pl


mnist = MnistParser()
knn = KNeighborsClassifier(n_neighbors=1)

print("Selecting 1000 samples randomly")
x, y = mnist.select(seed=555, shuffle=True, size=1000)

folds = [3, 10, 50, 100, 1000]
cross_error = []

for k in folds:

    print("Creating %s folds.." % k)
    x_folds, y_folds = mnist.k_fold(x, y, k)

    errors = []

    print("Computing %s-fold cross-validation error\n" % k)
    for selected in tqdm(range(len(x_folds))):
        x_test = x_folds.pop(selected)
        y_test = y_folds.pop(selected)
        x_train, y_train = [], []

        for e in x_folds:
            x_train += e

        for e in y_folds:
            y_train += e

        knn.fit(x_train, y_train)
        preds = knn.predict(x_test)

        real = np.array(y_test)

        wrong_preds = 0
        for idx, label in enumerate(real):
            if label[0] != preds[idx]:
                wrong_preds += 1

        errors.append(wrong_preds / len(preds))

        x_folds.insert(selected, x_test)
        y_folds.insert(selected, y_test)

    ce = sum(errors) / len(errors)
    cross_error.append(ce)

    print('\nCross validation accuracy : %s' % (1 - ce))


# Plot results
pl.plot(folds, cross_error, 'Cross validation error', 'Error', 'blue', 'Number of folds', 'Error value').show()