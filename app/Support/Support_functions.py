import csv
import numpy as np
import pandas as pd

"""
This file contains support functions for the machine learning methods.
The function OverfittingControl() controls if there is an overfitting situation.
The function ScaleData() is used in the methods that require a data scaling.
The function PerformanceOut() write on 2 output files the results of the process.
The function CompareParameters() is used for finding the best model.
The function SystemAnswer() gives to the user the prediction accordingly to the best model(s).
"""


def ScaleData(X_train, X_test, scaler):
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Converts the numpy array in DataFrame.
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    return X_train_scaled, X_test_scaled



def OverfittingControl(train_score, test_score):
    gap = float(train_score - test_score)
    if gap < 0.1: return True
    else: return False



def PerformanceOut(accuracy, confusion_matrix, classification_report,
                   prediction, probability, train_score, test_score, method:str, act:str):

    with open('Output/results.csv', act, newline='') as csvfile:
        field_names = ['Model', 'Accuracy', 'Prediction', 'Probability', 'Training Fit', 'Testing Fit', 'Fitting GAP']
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        # Columns title.
        if act == 'w':
            writer.writeheader()
        # Write the row with the data.
        writer.writerow({
            'Model': method,
            'Accuracy': accuracy,
            'Prediction': prediction,
            'Probability': probability,
            'Training Fit': train_score,
            'Testing Fit': test_score,
            'Fitting GAP': train_score - test_score
        })

    with open('Output/output.dat', act) as file:
        file.write('Accuracy {} : {}\n'.format(method, accuracy))
        file.write('Confusion Matrix {} :\n{}\n'.format(method, confusion_matrix))
        file.write('Classification Report {} :\n{}\n'.format(method, classification_report))
        file.write('--------------------------------------------------------\n')



def FillArrayParameters(key_par:str, report_knn, report_rf, report_lr, report_svm):
    # Save into an array the same parameters of the 4 methods.
    par_array = np.array([report_knn['weighted avg'][key_par],
                          report_rf['weighted avg'][key_par],
                          report_lr['weighted avg'][key_par],
                          report_svm['weighted avg'][key_par]])

    return par_array



def CompareParameters(f1_scores: np.ndarray, recalls: np.ndarray, precisions: np.ndarray,
                      matrices: np.ndarray, of_bools: np.ndarray):
    """
    Compare multiple models using F1-score, recall, precision,
    false negatives and false positives.

    Models in overfitting are excluded using the boolean mask.
    If all models are in overfitting, the comparison is done on all models.
    The function returns the indices of the best model(s).
    """
    # Step 0: filter models that are not in overfitting
    indices = np.where(of_bools)[0]

    # If all models are in overfitting, use all models.
    # Refill the indices np.array with all indices.
    if len(indices) == 0: indices = np.arange(len(f1_scores))

    # If only one model remains, return it
    elif len(indices) == 1: return indices

    # Step 1: F1-score (maximize)
    f1_sub = f1_scores[indices] # Take only the 'indices' position.
    best_f1 = np.where(f1_sub == np.max(f1_sub))[0] # Return an array of indices.
    # The next passage is useful for avoiding the indices mismatch.
    indices = indices[best_f1] # Update on 'indices' the elements at the positions in 'best_f1'.

    if len(indices) == 1: return indices

    # Step 2: Recall (maximize)
    recalls_sub = recalls[indices]
    best_recall = np.where(recalls_sub == np.max(recalls_sub))[0]
    indices = indices[best_recall]

    if len(indices) == 1: return indices

    # Step 3: Precision (maximize)
    precisions_sub = precisions[indices]
    best_precision = np.where(precisions_sub == np.max(precisions_sub))[0]
    indices = indices[best_precision]

    if len(indices) == 1: return indices

    # Step 4: False Negatives (minimize)
    fn_values = matrices[indices][:, 1, 0]
    best_fn = np.where(fn_values == np.min(fn_values))[0]
    indices = indices[best_fn]

    if len(indices) == 1: return indices

    # Step 5: False Positives (minimize)
    fp_values = matrices[indices][:, 0, 1]
    best_fp = np.where(fp_values == np.min(fp_values))[0]
    indices = indices[best_fp]

    return indices



def SystemAnswer(indices:np.ndarray, accuracy, f1_score, recall, precision, predict, prob, check):
    for i in range(np.size(indices)):
        model_name = str()
        match indices[i]:
            case 0: model_name = 'K-Neighbors'
            case 1: model_name = 'Random Forest'
            case 2: model_name = 'Logistic Regression'
            case 3: model_name = 'Space Vector Machine'

        action = 'w' if i == 0 else 'a'
        with open('Output/answer.dat', action) as f:
            f.write('MODEL : {}\n\n'.format(model_name))
            f.write('F1-SCORE : {}\nRECALL : {}\nPRECISION : {}\n'.format(f1_score[indices[i]], recall[indices[i]], precision[indices[i]]))
            f.write('ACCURACY : {}\n'.format(accuracy[i]))
            f.write('FIT : {}\n\n'.format('OVERFITTING' if not check[indices[i]] else 'OK'))
            f.write('PREDICTION : {}\n'.format('YES' if predict[indices[i]] == 1 else 'NO'))
            f.write('PROBABILITY : {}\n'.format(prob[indices[i]]))
            f.write('-----------------------------------------------\n')
