from mnistparser import MnistParser
from knn import KNeighborsClassifier
from tqdm import tqdm
import numpy as np
import plotter as pl

# Create a new instance of the parser
mnist = MnistParser()
# Create a new instance of the KNN model with K=1
knn = KNeighborsClassifier(n_neighbors=1)

# Use the parse to select randomly 1000 samples out of the 10000 available
print("Selecting 1000 samples randomly")
x, y = mnist.select(seed=555, shuffle=True, size=1000)

# Container of indices for k-fold cross validation to be performed
folds = [3, 10, 50, 100, 1000]

# Will contain the cross validation errors for each fold
cross_error = []

print("Cross validating with different fold sizes: " + str(folds))

# For each index in the container of indices
for k in folds:

    # Create a fold with the right K
    # x_folds and y_folds are arrays whose size is 1000 / k, where k is the size of the fold
    x_folds, y_folds = mnist.k_fold(x, y, k)

    # We have this array to save the temporary errors while performing cross validation for each fold
    errors = []
    # Select one of the folds sequentially
    # Use the selected one as test and the others as training samples
    # We use pop/insert macros to work on the fold arrays and ensure consistency between loops
    for selected in tqdm(range(len(x_folds))):
        x_test = x_folds.pop(selected)
        y_test = y_folds.pop(selected)
        x_train, y_train = [], []

        # Merge all the x folds in a single training vector
        for e in x_folds:
            x_train += e

        # Merge all the y folds in a single training vector
        for e in y_folds:
            y_train += e

        # Fit model
        knn.fit(x_train, y_train)

        # Predict label
        preds = knn.predict(x_test)

        # Convert the real labels to np array for convenience
        real = np.array(y_test)

        # Compute the error for the current fold
        wrong_preds = 0
        for idx, label in enumerate(real):
            if label != preds[idx]:
                wrong_preds += 1

        # Add the computed error to the temporary error container
        errors.append(wrong_preds / len(preds))

        # Reinsert the fold used for testing in the fold array
        x_folds.insert(selected, x_test)
        y_folds.insert(selected, y_test)

    # Before starting over with a different k value for a new cross validation computation
    # Compute the cross validation error (ce) and add it to the main cross validation error container
    ce = sum(errors) / len(errors)
    cross_error.append(ce)

# Print results
for idx, err in enumerate(cross_error):
    print('%s-fold cross validation: accuracy : %s | error : %s ' % (folds[idx], (1 - err), err))


# Plot results using a wrapper made around matplotlib
plt = pl.plot(folds, cross_error, 'Cross validation error', 'Error', 'blue', 'Number of folds', 'Error value')
plt.show()
