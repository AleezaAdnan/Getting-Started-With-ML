# Unsupervised Learning

Unsupervised Learning is the type of machine learning where the models are trained on unlabeled data (i.e without human supervision). They are able to identify patterns and derive insights without any guidance. 
For instance, Customer Segmentation based on purchasing behaviour 

There are three types of unsupervised learning tasks
- Clustering
- Association
- Dimensionality Reduction

## Clustering

Clustering is designed to group unlabeled data based on their similarities, data points within a cluster are more similar to each other than those outside the cluster.

Each data point can belong to one ( Hard Clustering ) on multiple clusters with different probabilities( Soft Clustering)

The foundation of clustering is the concept of similarity or distance between data points. The more similar two points are, the closer they are in terms of distance in the feature space, This distance could be euclidean or Manhattan etc depending on the nature of data and specific clustering algorithm being used.

### Types of Clustering
Clustering can be broadly categorized into several types based on how the data points are grouped:

i. **Partitioning Clustering** Divides the data into distinct, non-overlapping subsets (clusters).

ii. **Hierarchical Clustering** Builds a tree-like structure of clusters, either by merging smaller clusters into larger ones (agglomerative) or by splitting larger clusters into smaller ones (divisive).

iii. **Density-Based Clustering** Groups data points that are closely packed together, marking points that lie alone in low-density regions as outliers.

iv. **Model-Based Clustering** Assumes that the data is generated by a mixture of underlying probability distributions (e.g., Gaussian distributions) and assigns data points to clusters based on these distributions.

## Clustering Algorithms

### K-Means

K-Means is a partitioning algorithm that divides the dataset into K distinct, non-overlapping clusters. It works by iteratively refining the position of cluster centroids to minimize the overall variance within each cluster.

#### Working
1.  **Choose number of clusters** The first step is deciding how many clusters (K) you want to divide your data into. Initially you can experiment with different values to see what gives the optimal result, however there are various methods such as elbow method or the Sillhouette method for finding the optimal k value. 
2.  **Select centroids** At the start, K-Means randomly selects K data points from the dataset as the initial centroids. 
3.  **Assign datapoints to centroids** For each data point in the dataset, K-Means calculates the distance between the point and each of the K centroids.The data point is then assigned to the cluster whose centroid is closest. This step groups the data into K cluster
4.  **Update centroids** Once all data points have been assigned to clusters, K-Means recalculates the centroids of these clusters. The new centroid of a cluster is the mean (average) position of all the data points in that cluster, So the centroids better represent the center of their respective clusters.
5.  **Iterating** This process of reassogning datapoints and updating centroids continues until the centroids stop changing significantly, i.e the clusters have stabilized and the algorithm has converged.

K-means works best for problems with well seperated data, where clusters are approximately of the same density. It can cater large datasets and high dimensional data effectively.
However it's sensitive to outliers and the final clusters can be sensitive to the initial placement of centroids. 