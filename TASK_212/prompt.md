You are given a dataset containing a list of numerical values. Your goal is to detect and remove anomalies using the z-score method and then compute the average of the remaining values. The current notebook incorrectly calculates the variance instead of the standard deviation, and filters based on raw values instead of z-scores. As a result, the computed average differs from the expected one.



Fix the implementation so that it correctly:

Computes the mean and standard deviation.

Calculates each point’s z-score:

z=(x−mean)/std


Removes all points where |z| > 3.

Calculates the mean of the filtered dataset and stores it in a variable named average.

Save the corrected version as the final notebook.