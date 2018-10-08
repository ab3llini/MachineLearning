from irisparser import IrisParser
from knn import KNeighborsClassifier
import numpy as np
from matplotlib import pyplot as plt

# Mesh resolutions
resolution = 0.03

for df in ['iris_m10', 'iris_m20', 'iris_m30', 'iris_m50']:

    print('Working on dataset = ' + df)

    # Create a new instance of the parser
    iris = IrisParser(df)

    # Create a new instance of the KNN model with K=1
    knn = KNeighborsClassifier(n_neighbors=3)

    # Get features and labels
    features, labels = iris.parse(shuffle=False)

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
    c = lambda x: 'red' if x == 1 else 'blue' if x == 2 else 'green'
    mesh_colors = [c(p) for p in mesh_out]

    print("Drawing graph | total points = " + l)


    plt.title('Decision boundary for Iris')
    plt.ylabel('Feature 1')
    plt.xlabel('Feature 2')

    # Region colors

    plt.scatter(mesh_in[:, 0], mesh_in[:, 1], color=mesh_colors, s=1)

    plt.show()
