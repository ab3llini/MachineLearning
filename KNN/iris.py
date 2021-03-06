from irisparser import IrisParser
from knn import KNeighborsClassifier
from matplotlib import pyplot as plt
from cross_validate import *
import plotter as plotter
from matplotlib import colors as mcolors

# Better colors mapping for pyplot
colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

# Mesh resolutions
resolution = 0.02

# This array holds the loocv errors for the various datasets
loocvs = []

# This array holds the loocv accuracies for the various datasets
accuracies = []

dsets = ['iris_m10', 'iris_m20', 'iris_m30', 'iris_m50']

for df in dsets:

    print('Working on dataset = ' + df)

    # Create a new instance of the parser
    iris = IrisParser(df)

    # Create a new instance of the KNN model with K=1
    knn = KNeighborsClassifier(n_neighbors=3)

    # Get features and labels
    features, labels = iris.parse(shuffle=False)

    # ************************************************* Train error
    print('Computing training LOOCV error')

    # Create LOOCV folds
    x_folds, y_folds = iris.k_fold(features, labels, len(features))

    # Compute LOOCV error
    loocv_e = cross_validate(x_folds, y_folds, knn)

    # Add this error to a list, for later plotting, and the accuracy too
    loocvs.append(loocv_e)
    accuracies.append(1 - loocv_e)

    print('loocv training error for dataset %s: accuracy : %s | error : %s ' % (df, (1 - loocv_e), loocv_e))

    features = np.array(features)
    labels = np.array(labels)

    # We have 2D features, lets call one axis x and the other y
    x = features[:, 0]
    y = features[:, 1]

    # Fit the model with the dataset
    knn.fit(features, labels)

    # Get max and min values per axis and build a mesh for plotting the boundary
    min_x, min_y, max_x, max_y = min(x), min(y), max(x), max(y)

    print("Building mesh points | resolution = %s" % resolution)

    # Matrix representing our mesh input data and output labels
    mesh_in = []

    # Build the mesh
    for step_x in np.arange(min_x, max_x, resolution):
        for step_y in np.arange(min_y, max_y, resolution):
            mesh_in.append([step_x, step_y])

    # Convert to numpy for performance reasons
    mesh_in = np.array(mesh_in)

    # Single pass to count elements used for debug
    l = str(len(mesh_in))

    print('Predicting.. | total points to predict = %s | This might take some time..' % l)
    mesh_out = np.array(knn.predict(mesh_in))

    print('Computing mesh colors..')

    c = lambda x: 'turquoise' if x == 1 else 'gold' if x == 2 else 'dodgerblue'

    # Prepare the mesh colors
    mesh_colors = [colors[c(p)] for p in mesh_out]
    features_colors = [colors[c(p)] for p in labels]

    print("Drawing graph | total points = " + l)

    # Draw the decision boundaries
    fig = plt.figure()
    plt.title('Decision boundary for ' + df)
    plt.ylabel('Feature 1')
    plt.xlabel('Feature 2')
    # Draw the mesh
    plt.scatter(mesh_in[:, 0], mesh_in[:, 1], color=mesh_colors, s=2)

    # Draw the dataset points
    plt.scatter(x, y, color=features_colors, s=40, edgecolors='k')


    plt.show()
    name = df+'.png'
    fig.savefig(name, dpi=300)


# Plot loocv errors for different datasets
plotter.plot(dsets, loocvs, title='Train errors', legend='loocv error', color='blue', xlabel='dataset', ylabel='error', fname='iris_loocv_err')
plotter.plot(dsets, accuracies, title='Train accuracies', legend='loocv accuracy', color='green', xlabel='dataset', ylabel='accuracy', fname='iris_loocv_acc')