from tqdm import tqdm
import numpy as np


def cross_validate(x_folds, y_folds, model):

    x_folds = list(x_folds)
    y_folds = list(y_folds)

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
            model.fit(x_train, y_train)

        # Predict label
        preds = model.predict(x_test)

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
    return ce