You’re training a crop disease classification model and need to complete the evaluation with correct cross-validation and metrics.

Perform the following updates:

Fix cross-validation folds:

Change cv=3 to cv=5.

Store CV results:

Save all 5 accuracy scores in cv_scores (NumPy array).

Store the mean in cv_mean_score and std in cv_std_score.

Add stratified cross-validation:

Use StratifiedKFold with 5 splits.

Save results in stratified_cv_scores, mean in stratified_cv_mean, and std in stratified_cv_std.

Add F1 and ROC-AUC metrics:

Store F1 scores in cv_f1_scores and mean in cv_f1_mean.

Store ROC-AUC scores in cv_roc_auc_scores and mean in cv_roc_auc_mean.