from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from app.Support.Support_functions import *

"""
In this file we use the KNeighbors method for make a prediction.
The method takes as input the train and test values, the K value and the value of an external point.
At the end of the method, the process write the result of the prediction with PerformanceOut().
"""

def KNeighbors_prediction(X_train, X_test, Y_train, Y_test, scaler, Neighbors:int, gender, age:int, salary:float):
    # Scale the data with the method ScaleData().
    X_train_scaled, X_test_scaled = ScaleData(X_train, X_test, scaler)
    # Create the model KNeighbors.
    knn = KNeighborsClassifier(n_neighbors=Neighbors)
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

    return predict[0], probability[0], accuracy, confusionMatrix, report, check_overfitting

