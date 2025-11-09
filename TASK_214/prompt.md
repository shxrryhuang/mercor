You are given a dataset of tuples in the form (category, value) representing values across multiple groups.
The goal is to compute the mean value for each category.

The current notebook incorrectly sums the values but never divides by the count of items per category, leading to wrong results.
Fix the code so that for each category:

You correctly compute the total sum and count of its values.

You divide the total by the count to obtain the mean.

Store the final dictionary in a variable called group_means.

The output should show each category’s mean value correctly.