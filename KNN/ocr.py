from mnistparser import MnistParser
from knn import KNeighborsClassifier
from cross_validate import *
import plotter as pl

# Create a new instance of the parser
mnist = MnistParser()
# Create a new instance of the KNN model with K=1
knn = KNeighborsClassifier(n_neighbors=1)

# ************************************************* Train error
# Plot of test error with different training size
tr_sizes = [100, 1000, 2500, 5000, 7500, 10000]
loocv_errors = []

print('Computing training LOOCV error with this training sizes: ' + str(tr_sizes))

for s in tr_sizes:

    # Select s samples randomly
    x, y = mnist.select(seed=555, shuffle=True, size=s)

    # Create LOOCV folds
    x_folds, y_folds = mnist.k_fold(x, y, s)

    # Compute LOOCV error
    loocv_errors.append(cross_validate(x_folds, y_folds, knn))

for idx, err in enumerate(loocv_errors):
    print('loocv training error with %s samples: accuracy : %s | error : %s ' % (tr_sizes[idx], (1 - err), err))


# Print errors
pl.plot(tr_sizes, loocv_errors, title='Train errors', legend='loocv error', color='blue', xlabel='train size', ylabel='error', fname='ocr_loocv')

# ************************************************* Cross validation error
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

    cross_error.append(cross_validate(x_folds, y_folds, knn))

# Print results
for idx, err in enumerate(cross_error):
    print('%s-fold cross validation: accuracy : %s | error : %s ' % (folds[idx], (1 - err), err))


# Plot results using a wrapper made around matplotlib
plt = pl.plot(folds, cross_error, 'Cross validation error', 'Error', 'blue', 'Number of folds', 'Error value')
plt.show()
