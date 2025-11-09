You are a machine learning engineer working on a customer segmentation task for an e-commerce company. The current notebook runs KMeans clustering, but it only prints the cluster labels and does not evaluate cluster quality or interpret the clusters properly. You must enhance the notebook with multiple cluster evaluation metrics and additional variables for interpretation.

Specifically, you must:
Fix KMeans clustering
Use n_clusters=4 (currently missing).
Use random_state=42 for reproducibility.
Store fitted model in a variable named kmeans_model.


Add evaluation metrics
Compute and store:
inertia_value = KMeans model’s inertia
silhouette_avg = silhouette score across all samples
davies_bouldin = Davies–Bouldin index (lower is better)


Add cluster interpretation
Create a DataFrame cluster_summary containing:
One row per cluster
Mean values for each feature per cluster
Sort clusters by their mean spending score (spending_score column).


All new variables must be clearly defined and printed.


