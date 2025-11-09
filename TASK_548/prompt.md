You’re working on a heart disease prediction model and need to fix and complete its cross-validation evaluation.

Implement the following fixes in the notebook:

Fix incorrect CV folds:

Current setup uses only cv=3. Change it to cv=5 for more robust evaluation.

Save cross-validation results:

Store all 5 accuracy scores from cross_val_score() in cv_scores (NumPy array).

Compute mean (cv_mean_score) and standard deviation (cv_std_score).

Add stratified cross-validation:

Use StratifiedKFold with 5 folds to maintain class balance.

Store results in stratified_cv_scores (NumPy array), mean in stratified_cv_mean, and std in stratified_cv_std.

Evaluate additional metrics:

Add F1-score CV: store results in cv_f1_scores and mean in cv_f1_mean.

Add ROC-AUC CV: store results in cv_roc_auc_scores and mean in cv_roc_auc_mean.

After the fixes, the notebook must contain:

Correct 5-fold cross-validation with accuracy scores and summary stats

Stratified cross-validation results

F1 and ROC-AUC cross-validation arrays and mean values