from mnistparser import MnistParser
from knn import *
from cross_validate import cross_validate
import plotter as pl


# Remap
parse_target = {
    'training': 0,
    'testing': 1
}


# Create new instances of the parser, one for training and one for testing
tr_parser = MnistParser(target=parse_target['training'])
ts_parser = MnistParser(target=parse_target['testing'])

# Create a new instance of the KNN model with K=1
knn = KNeighborsClassifier(n_neighbors=1)


# Used to compute how the classification error for the 1-Nearest Neighbor
# on the MNIST dataset change with the number of training examples
# This takes approx 30min, it has already been computed and the results are shown in the "ocr_train_loocv_err.png" image
# This is a general p
def classification_error(seed=555):

    nr_training_examples = [100, 1000, 2500, 5000, 7500, 10000]
    errors = []
    accuracies = []

    print('Computing classification error.. be patient.')

    for s in nr_training_examples:

        # Select s samples randomly from the training set and uses them to fit the knn model
        x_tr, y_tr = tr_parser.select(seed=seed, shuffle=True, size=s)

        # Select all the testing samples
        x_ts, y_ts = ts_parser.select(shuffle=False, size=10000)

        # Fit the model with the current training set
        knn.fit(x_tr, y_tr)

        # Predict the testing set
        preds = knn.predict(x_ts)

        # Compute the error for the current fold
        wrong_preds = 0
        for idx, label in enumerate(y_ts):
            if label != preds[idx]:
                wrong_preds += 1

        error = wrong_preds / len(preds)
        errors.append(error)
        accuracies.append(1 - error)

        print('Classification error with %s samples: accuracy : %s | error : %s ' % (s, (1 - error), error))

    # Print errors & accuracies
    pl.plot(nr_training_examples, errors, title='Classification error', legend='error', color='blue', xlabel='Train size', ylabel='Error', fname='ocr_test_loocv_err')
    pl.plot(nr_training_examples, accuracies, title='Classification accuracy', legend='accuracy', color='green', xlabel='Train size', ylabel='Accuracy', fname='ocr_test_loocv_acc')


# Cross validation error, computed on the training set
def n_fold_cross_val(seed=555):

    # Use the parser to select randomly 1000 samples out of the 10000 available
    print("Selecting 1000 samples randomly")
    x, y = tr_parser.select(seed=seed, shuffle=True, size=1000)

    # Container of indices for k-fold cross validation to be performed
    folds = [3, 10, 50, 100, 1000]

    # Will contain the cross validation errors for each fold
    cross_error = []

    print("Cross validating with different fold sizes: " + str(folds))

    # For each index in the container of indices
    for k in folds:

        # Create a fold with the right K
        # x_folds and y_folds are arrays whose size is 1000 / k, where k is the size of the fold
        x_folds, y_folds = tr_parser.k_fold(x, y, k)

        cross_error.append(cross_validate(x_folds, y_folds, knn))

    # Print results
    for idx, err in enumerate(cross_error):
        print('%s-fold cross validation: accuracy : %s | error : %s ' % (folds[idx], (1 - err), err))

    # Plot results using a wrapper made around matplotlib
    plt = pl.plot(folds, cross_error, 'Cross validation error', 'Error', 'blue', 'Number of folds', 'Error value')
    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # IMPORTANT
# High complexity, approx 30 min required, already pre computed, output in ocr_train_loocv_err.png file
# classification_error(seed=555)

# Run cross validation
n_fold_cross_val(seed=555)
