Given a dataset of tuples (category, score, weight),
calculate the weighted average score only for a specific category (e.g., "A").

The initial notebook incorrectly calculates the unweighted mean or includes wrong categories.

You must fix it to:

Filter only the chosen category ("A").

Compute weighted_avg = sum(score * weight) / sum(weight).