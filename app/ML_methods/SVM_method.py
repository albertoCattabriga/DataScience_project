from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from app.Support.Support_functions import *

"""
In this file we use the Support Vector Machine (SVM) method for make a prediction.
The SVM finds the optimal hyperplane that separates the two classes
with the maximum margin.
"""

def SVM_prediction(X_train, X_test, Y_train, Y_test, scaler, gender, age:int, salary:float):
    # Scale the data with the method ScaleData().
    X_train_scaled, X_test_scaled = ScaleData(X_train, X_test, scaler)

    # Search the best kernel.
    set_kernel = {'kernel' : ['rbf', 'linear']}
    search_kernel = GridSearchCV(SVC(random_state=0), set_kernel, cv=6, n_jobs=-1, scoring='f1_weighted')

    # Find the best kernel.
    search_kernel.fit(X_train_scaled, Y_train)
    best_kernel = search_kernel.best_params_['kernel']

    print('SVM starting with kernel : {} ...'.format(best_kernel))

    # Create the model SVM.
    # kernel='rbf' handles non-linear boundaries (better for this dataset).
    # probability=True allows predict_proba().
    svm = SVC(kernel=best_kernel, probability=True, random_state=0)

    # Train the model.
    svm.fit(X_train_scaled, Y_train)

    # Start the validation part.
    y_predict = svm.predict(X_test_scaled)

    # Evaluation of the performance.
    accuracy = accuracy_score(Y_test, y_predict)
    confusionMatrix = confusion_matrix(Y_test, y_predict)
    report = classification_report(Y_test, y_predict, output_dict=True)
    train_score = svm.score(X_train_scaled, Y_train)
    test_score = svm.score(X_test_scaled, Y_test)

    # Prediction of an external point.
    point_array = pd.DataFrame([[int(gender), int(age), float(salary)]],
                               columns=['Gender', 'Age', 'EstimatedSalary'])
    # Transform the new data in a consistent format.
    df_scaled = scaler.transform(point_array)
    # Put the transformed data in a new dataframe with consistent data format.
    df_scaled = pd.DataFrame(df_scaled, columns=['Gender', 'Age', 'EstimatedSalary'])

    # Final classification.
    predict = svm.predict(df_scaled)
    probability = svm.predict_proba(df_scaled) * 100

    # Write results on the output files.
    PerformanceOut(accuracy, confusionMatrix, classification_report(Y_test, y_predict),
                   predict[0], probability[0], train_score, test_score, method='SVM', act='a')

    check_overfitting = OverfittingControl(train_score, test_score)

    print('SVM completed!')

    return predict[0], probability[0], accuracy, confusionMatrix, report, check_overfitting