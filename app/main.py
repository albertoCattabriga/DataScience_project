from app.Support.Support_functions import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from ML_methods.KNN_method import KNeighbors_prediction
from ML_methods.RandomForest_method import RandomForest_prediction
from ML_methods.LogisticRegression_method import LogisticRegression_prediction
from ML_methods.SVM_method import  SVM_prediction

"""
This is the main.py file of the structure.
This file, first of all, reads the dataset from the file 'Social_Network_Ads.csv' and takes the 
info about the gender, age and estimated salary of the people.
When all the data are saved, it runs all the methods in files .py into 'ML_methods', prints the 
results and see what methods is the best and why. 
"""

def main():
    # Read the file .csv
    df_raw = pd.read_csv('../Social_Network_Ads.csv')
    # Discard the column of User ID.
    df = df_raw.drop(['User ID'], axis=1)
    # The 'gender' should be mapped because it needs numeric value.
    df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})

    # Split the training values and the test values.
    X_train, X_test, Y_train, Y_test = train_test_split(df.drop(['Purchased'], axis=1),
                                                        df['Purchased'],
                                                        test_size=0.2,
                                                        random_state=0)

    # Create a StandardScaler object.
    scaler = StandardScaler()

    # Now everything is ready for make prediction by using all the methods.
    # Save the data of the external case.
    gender = int()
    gender_str = input('Write the gender :\t')
    if gender_str.lower() != 'male' and gender_str.lower() != 'female':
        print('ERROR : Gender not admissible!')
        exit()
    if gender_str.lower() == 'male': gender = 0
    else: gender = 1

    age = int(input('Digit the age:\t'))
    salary = int(input('Digit the salary:\t'))

    # Remember: Every method return predict, probability, accuracy, confusionMatrix, report and check ofr overfitting.
    """
    Prediction with KNN.
    """
    predict_knn, prob_knn, accuracy_knn, confMatrix_knn, report_knn, overfitting_knn = KNeighbors_prediction(X_train, X_test,
                                                                                            Y_train, Y_test,
                                                                                            scaler, 5,
                                                                                            gender, age, salary)

    """
    Prediction with Random Forest.
    """
    predict_rf, prob_rf, accuracy_rf, confMatrix_rf, report_rf, overfitting_rf = RandomForest_prediction(X_train, X_test,
                                                                                         Y_train, Y_test,
                                                                                         100, 8,
                                                                                         gender, age, salary)

    """
    Prediction with Logistic Regression.
    """
    predict_lr, prob_lr, accuracy_lr, confMatrix_lr, report_lr, overfitting_lr = LogisticRegression_prediction(X_train, X_test,
                                                                                               Y_train, Y_test,
                                                                                               scaler, gender, age, salary)

    """
    Prediction with Space Vector Machine.
    """
    predict_svm, prob_svm, accuracy_svm, confMatrix_svm, report_svm, overfitting_svm = SVM_prediction(X_train, X_test, Y_train, Y_test,
                                                                                     scaler, gender, age, salary)


    """
    Once all the model are completed, start the evaluation of the performance.
    """
    # Save the parameters into arrays.
    # F1-SCORE.
    f1_scores = FillArrayParameters('f1-score', report_knn, report_rf, report_lr, report_svm)
    # RECALL.
    recalls = FillArrayParameters('recall', report_knn, report_rf, report_lr, report_svm)
    # PRECISION.
    precisions = FillArrayParameters('precision', report_knn, report_rf, report_lr, report_svm)
    # CONFUSION MATRIX.
    matrices = np.array([confMatrix_knn, confMatrix_rf, confMatrix_lr, confMatrix_svm])
    # OVERFITTING BOOLS.
    overfitting_checks = np.array([overfitting_knn, overfitting_rf, overfitting_lr, overfitting_svm])

    # Start the comparison between models.
    index_best_model = CompareParameters(f1_scores, recalls, precisions, matrices, overfitting_checks)

    """
    Communicate with 'response.dat' the best answer of the system with:
    - Best model and its parameters.
    - Prediction.
    - Probability.
    """
    # Save the prediction and the probability into 2 arrays.
    predicts = [predict_knn, predict_rf, predict_lr, predict_svm]
    probs = [prob_knn, prob_rf, prob_lr, prob_svm]
    accuracies = [accuracy_knn, accuracy_rf, accuracy_lr, accuracy_svm]

    # Communicate the answer to the user.
    SystemAnswer(index_best_model, accuracies, f1_scores, recalls, precisions, predicts, probs, overfitting_checks)


if __name__ == '__main__':
    main()