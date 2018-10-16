from mnistparser import MnistParser
from knn import *
from cross_validate import cross_validate
import plotter as pl


# Remap
parse_target = {
    'training': 0,
    'testing': 1
}


# Create a new instance of the parser
mnist_train = MnistParser(target=parse_target['training'])
mnist_test = MnistParser(target=parse_target['testing'])

# Create a new instance of the KNN model with K=1
knn = KNeighborsClassifier(n_neighbors=1)


# Used to compute errors on train/test set with LOOCV
# This takes approx 30min, it has already been computed and the results are shown in the "ocr_train_loocv_err.png" image
# This is a general p
def compute_error(data, seed=555):
    # Plot of test error with different training size
    sizes = [100, 1000, 2500, 5000, 7500, 10000]
    loocv_errors = []
    loocv_accuracies = []

    print('Computing LOOCV error with this sizes: ' + str(sizes))

    for s in sizes:

        # Select s samples randomly
        x, y = data.select(seed=seed, shuffle=True, size=s)

        # Create LOOCV folds
        x_folds, y_folds = data.k_fold(x, y, s)

        # Compute LOOCV error
        e = cross_validate(x_folds, y_folds, knn)

        # Add errors & accuracy to array
        loocv_errors.append(e)
        loocv_accuracies.append(1 - e)

    for idx, err in enumerate(loocv_errors):
        print('loocv error with %s samples: accuracy : %s | error : %s ' % (sizes[idx], (1 - err), err))

    # Print errors & accuracies
    pl.plot(sizes, loocv_errors, title='Classification error', legend='loocv error', color='blue', xlabel='Test size', ylabel='Error', fname='ocr_test_loocv_err')
    pl.plot(sizes, loocv_accuracies, title='Classification accuracy', legend='loocv accuracy', color='green', xlabel='Test size', ylabel='Accuracy', fname='ocr_test_loocv_acc')


# Cross validation error, computed on the training set
def cv(seed=555):

    # Use the parser to select randomly 1000 samples out of the 10000 available
    print("Selecting 1000 samples randomly")
    x, y = mnist_train.select(seed=seed, shuffle=True, size=1000)

    # Container of indices for k-fold cross validation to be performed
    folds = [3, 10, 50, 100, 1000]

    # Will contain the cross validation errors for each fold
    cross_error = []

    print("Cross validating with different fold sizes: " + str(folds))

    # For each index in the container of indices
    for k in folds:

        # Create a fold with the right K
        # x_folds and y_folds are arrays whose size is 1000 / k, where k is the size of the fold
        x_folds, y_folds = mnist_train.k_fold(x, y, k)

        cross_error.append(cross_validate(x_folds, y_folds, knn))

    # Print results
    for idx, err in enumerate(cross_error):
        print('%s-fold cross validation: accuracy : %s | error : %s ' % (folds[idx], (1 - err), err))

    # Plot results using a wrapper made around matplotlib
    plt = pl.plot(folds, cross_error, 'Cross validation error', 'Error', 'blue', 'Number of folds', 'Error value')
    plt.show()


# High complexity, approx 30 min required, already pre computed, output in ocr_train_loocv_err.png file
# compute_error(data=mnist_train, seed=555)
compute_error(data=mnist_test, seed=555)


# Run cross validation
cv(seed=555)