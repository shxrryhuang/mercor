You are analyzing product-quality consistency across multiple manufacturing factories.
The provided Jupyter Notebook already loads a dataset with columns "factory" and "quality_score", but your teammate accidentally computed the mean per factory instead of the standard deviation.

Your task is to correct the analysis by:

Grouping the data by "factory" and computing the standard deviation of "quality_score" for each factory.
Store the result in a Series named factory_stats.

Determine which factory has the highest variability in quality:

highest_std_factory = factory_stats.idxmax()
highest_std_value = factory_stats.max()


Print the per-factory standard deviations and the factory with the highest variability.

Validation requirements:

The notebook executes successfully.

factory_stats must contain the correct standard deviations.

highest_std_factory must equal the factory with the largest std value.

The test verifies numeric accuracy against a reference computation.