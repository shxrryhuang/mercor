You are working as a data scientist for a telecommunications company analyzing customer churn prediction. A colleague prepared a notebook using a logistic regression model, but their evaluation setup is incomplete and not statistically robust. Your task is to fix and extend the model evaluation pipeline to include proper cross-validation and model comparison between Logistic Regression and Random Forest.

Specifically, you must:

1.Fix cross-validation:

Use 5-fold cross-validation instead of 3-fold for better evaluation reliability. Store all 5 fold accuracy scores in a NumPy array variable called cv_accuracy_scores.
Compute and store:
Mean accuracy as cv_accuracy_mean
Standard deviation as cv_accuracy_std
2.Add Random Forest comparison:

Train a RandomForestClassifier (n_estimators=100, random_state=42).
Evaluate it using 5-fold CV and store results in:
rf_cv_scores (NumPy array)
rf_cv_mean
rf_cv_std

3.Add stratified validation:

Use StratifiedKFold(n_splits=5, shuffle=True, random_state=42) to preserve class balance.
Evaluate both models under this stratified scheme and store:
Logistic Regression stratified scores: stratified_lr_scores
Random Forest stratified scores: stratified_rf_scores
4.Print and compare all metrics clearly at the end.