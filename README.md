#Data Science - Binary classification

This project presents a comparative analysis of multiple supervised machine learn-
ing classification algorithms applied to the “ Social Network Ads” dataset. The
goal is to predict whether a user is likely to purchase a product based on demo-
graphic features such as gender, age and estimated salary.

Four classification models were implemented and evaluated: K-Nearest Neigh-
bors (KNN), Random Forest, Logistic Regression and Support Vector
Machine (SVM). Each model was trained on the same split of the dataset and
assessed using standard performance metrics, including accuracy, precision, recall,
F1-score, and confusion matrix. An overfitting control was also included to ensure
the reliability of each model’s generalisation capability.

The project follows a modular architecture: data preprocessing, model training,
prediction, and evaluation are each encapsulated in dedicated modules, promoting
code clarity and reusability. At the end of the pipeline, an automated comparison
selects the best-performing model and reports its prediction along with the asso-
ciated probability for a given input case.

The data set used is the publicly available “ Social Network Ads” dataset, which
contains user records along with a binary label indicating whether they made a
purchase following exposure to advertisements.
