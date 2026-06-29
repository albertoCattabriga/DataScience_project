from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from app.Support.Support_functions import *
import matplotlib.pyplot as mplt
import numpy as np

"""
In this file we use the KNeighbors method for make a prediction.
The method takes as input the train and test values, the K value and the value of an external point.
At the end of the method, the process write the result of the prediction with PerformanceOut().
"""

def _plot_knn_3d(X_train_scaled, Y_train, df_scaled, predict, knn, k):
    """
    Visualize the K-neighbors of the query point.
    """
    # Converts in Numpy array.
    X_arr = np.array(X_train_scaled)
    Y_arr = np.array(Y_train)
    # Extract the query point as 1D np.array().
    q = np.array(df_scaled)[0]

    # Find distances, indices and the K-neighbors to query point.
    distances, indices = knn.kneighbors(df_scaled)
    # Extract the coordinates of the k-neighbors.
    neighbor_coords = X_arr[indices[0]]  # shape: (k, 3)

    # Separate the two class, bool array.
    # Map which has 0 or 1.
    mask_0 = (Y_arr == 0)  # Not purchased
    mask_1 = (Y_arr == 1)  # Purchased
    # Label for the query point.
    label = 'Purchased' if predict == 1 else 'Not purchased'

    # Create 3D figure.
    figure = mplt.figure(figsize=(10, 7))
    ax = figure.add_subplot(111, projection='3d')

    # Scatter of 'NOT PURCHASED' points in red.
    # 0 -> gender, 1 -> age, 2-> EstimatedSalary.
    # In a 3D plot, scatter attends 3 coordinates.
    ax.scatter(X_arr[mask_0, 0], X_arr[mask_0, 1], X_arr[mask_0, 2],
               c='red', s=15, alpha=0.5, label='Not purchased')

    # Scatter of 'PURCHASED' points.
    ax.scatter(X_arr[mask_1, 0], X_arr[mask_1, 1], X_arr[mask_1, 2],
               c='green', s=15, alpha=0.5, label='Purchased')

    # Scatter query point (star shape).
    # * unpacked the value of the array 'q'.
    ax.scatter(*q, c='blue', s=120, marker='*', zorder=5,
               edgecolors='black', linewidths=1, label=f'Query point ({label})')

    # Red segments that links the query point with the neighbors.
    for i, nb in enumerate(neighbor_coords):
        ax.plot([q[0], nb[0]], [q[1], nb[1]], [q[2], nb[2]],
                color='black', linewidth=1.5, linestyle='--',
                # Add labels.
                label='K-nearest links' if i == 0 else None)

    # Axis labels.
    ax.set_xlabel('Gender (scaled)')
    ax.set_ylabel('Age (scaled)')
    ax.set_zlabel('Salary (scaled)')
    ax.set_title(f'KNN 3D — K={k} | Prediction class: {label}')
    ax.legend(loc='upper left')

    # Adjust automatically the spacing.
    mplt.tight_layout()
    mplt.show()


def KNeighbors_prediction(X_train, X_test, Y_train, Y_test, scaler, gender, age:int, salary:float):
    # Scale the data with the method ScaleData().
    X_train_scaled, X_test_scaled = ScaleData(X_train, X_test, scaler)

    # Find the ideal value of neighbors.
    set_K = {'n_neighbors' : list(range(1, 10))}
    # Use GridSearchCV for searching the best int values between the set of K.
    # Put in default the value into the argument n_neighbors of KNeighborsClassifier().
    # cv=6 makes a K-fold validation with K = 6.
    search_K = GridSearchCV(KNeighborsClassifier(algorithm='brute'), set_K, cv=6, n_jobs=-1, scoring='f1_weighted')
    search_K.fit(X_train_scaled, Y_train)

    # Save the best K.
    best_K = search_K.best_params_['n_neighbors']
    print(f'Starting KNN prediction with K : {best_K} ...')

    # Create the model KNeighbors with best K found.
    # # algorithm=brute means that it calculates all the distances between points.
    knn = KNeighborsClassifier(n_neighbors=best_K, algorithm='brute')
    # Fit the training data.
    knn.fit(X_train_scaled, Y_train)
    # Make the prediction.
    y_predict = knn.predict(X_test_scaled)

    # Evaluate the performance.
    accuracy = accuracy_score(Y_test, y_predict)
    confusionMatrix = confusion_matrix(Y_test, y_predict)
    report = classification_report(Y_test, y_predict, output_dict=True)
    train_score = knn.score(X_train_scaled, Y_train)
    test_score = knn.score(X_test_scaled, Y_test)

    # Make the prediction of an external point.
    point_array = pd.DataFrame([[int(gender), int(age), float(salary)]],
                               columns=['Gender', 'Age', 'EstimatedSalary'],)
    # Transform the new array in a consistent format of the data.
    df_scaled = scaler.transform(point_array)
    df_scaled = pd.DataFrame(df_scaled, columns=['Gender', 'Age', 'EstimatedSalary'])

    # Final classification of the external point.
    predict = knn.predict(df_scaled)
    probability = knn.predict_proba(df_scaled) * 100

    # Output on the file.
    PerformanceOut(accuracy, confusionMatrix, classification_report(Y_test, y_predict),
                   predict[0], probability[0], train_score, test_score, method='KNN', act='w')

    check_overfitting = OverfittingControl(train_score, test_score)

    print('KNN completed!')

    # Data visualization.
    try:
        _plot_knn_3d(X_train_scaled, Y_train, df_scaled, predict[0], knn, best_K)
    except Exception as e:
        print(f'ERROR : Impossible to reproduce the KNN graph, cause : {e}')

    return predict[0], probability[0], accuracy, confusionMatrix, report, check_overfitting

