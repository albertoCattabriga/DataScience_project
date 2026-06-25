from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from app.Support.Support_functions import *

"""
In this file we use the Random forest method for make a prediction.
The method takes as input the train and test values and the value of an external point.
At the end of the method, the process write the result of the prediction with PerformanceOut().
The code shows also the trees used for the Random forest classification with 'matplotlib'.
"""

def RandomForest_prediction(X_train, X_test, Y_train, Y_test, gender, age:int, salary:float):
    # In the random forest method, the data don't need to be scaled.

    # Find the best number of estimator and the best max depth.
    set_par = {'n_estimators' : list(range(50, 250, 50)),
                     'max_depth' : list(range(1, 10))}

    search_par = GridSearchCV(RandomForestClassifier(), set_par, cv=5, n_jobs=-1,
                               scoring='f1_weighted')

    ideal_par = search_par.fit(X_train, Y_train)
    best_estimators = ideal_par.best_params_['n_estimators']
    best_maxDepth = ideal_par.best_params_['max_depth']

    print('Starting Random Forest : n_estimators = {}, max_depth = {}'.format(best_estimators, best_maxDepth))

    # n_estimators: Number of trees for the prediction.
    # max_depth: Maximum level of the trees.
    random_forest = RandomForestClassifier(n_estimators=best_estimators, max_depth=best_maxDepth)
    # Train the model.
    random_forest.fit(X_train, Y_train)

    # Make a prediction.
    y_predict = random_forest.predict(X_test)

    # Save the performance parameters.
    accuracy = accuracy_score(Y_test, y_predict)
    confusionMatrix = confusion_matrix(Y_test, y_predict)
    report = classification_report(Y_test, y_predict, output_dict=True)
    train_score = random_forest.score(X_train, Y_train)
    test_score = random_forest.score(X_test, Y_test)

    # Classify the external point.
    point_array = pd.DataFrame([[int(gender), int(age), float(salary)]],
                               columns=['Gender', 'Age', 'EstimatedSalary'])

    # Final classification.
    predict = random_forest.predict(point_array)
    probability = random_forest.predict_proba(point_array) * 100

    # Output on the file.
    PerformanceOut(accuracy, confusionMatrix, classification_report(Y_test, y_predict),
                   predict[0], probability[0], train_score, test_score, method='RandomForest', act='a')

    check_overfitting = OverfittingControl(train_score, test_score)

    print('Random Forest completed!')

    """
    Use Matplotlib for print the trees.
    fig -> the entire figure that contains all the subplots.
    axes -> array of subplots, one for each tree to be shown (axes[0], axes[1], axes[2]...).
    """

    # Ask the user if and how many trees he wants to see.
    n_of_trees = int(input('How many trees would you like to see?\t'))
    if n_of_trees > 0:
        fig, axes = plt.subplots(nrows=1, ncols=n_of_trees, figsize=(30, 10))
        for i in range(n_of_trees):
            plot_tree(random_forest.estimators_[i],
                      feature_names=['Gender', 'Age', 'EstimatedSalary'],
                      class_names=['Not Purchased', 'Purchased'],
                      filled=True,
                      ax=axes[i], )
            axes[i].set_title(f'Tree {i + 1}')

        plt.show()

    return predict[0], probability[0], accuracy, confusionMatrix, report, check_overfitting