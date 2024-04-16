import numpy as np
import matplotlib.pyplot as plt

# Generate training data
np.random.seed(0)
X1 = np.random.multivariate_normal([2, 1], np.diag([0.4, 0.04]), 100)
X2 = np.random.multivariate_normal([1, 2], np.diag([0.4, 0.04]), 100)
X = np.concatenate((X1, X2), axis=0)

# Perform K-means
def kmeans(X, k, max_iter=100):
    # Randomly chose the centroids
    centroids = X[np.random.choice(X.shape[0], k, replace=False), :]

    for _ in range(max_iter):
        # Assign each data point to the closest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        cluster_assignments = np.argmin(distances, axis=0)

        # Update centroids based on the cluster assignments
        new_centroids = np.array([X[cluster_assignments == i].mean(axis=0) for i in range(k)])

        if np.all(centroids == new_centroids):
            break

        centroids = new_centroids

    return centroids, cluster_assignments

# Run kmeans
centroids, cluster_assignments = kmeans(X, k=2)

# Draw training points and centroids
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments)
plt.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='x')
plt.title('K-means Clustering (k=2)')
plt.show()

# Calculate K-means objective function
distances = np.sqrt(((X - centroids[cluster_assignments])**2).sum(axis=1))
objective_function = np.sum(distances**2)
print(f'K-means Objective Function Value: {objective_function:.2f}')

# Plot and print objective function for the first 5 iterations
max_iter = 5
centroids, cluster_assignments = kmeans(X, k=2, max_iter=max_iter)

for i in range(max_iter):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='r', marker='x')
    plt.title(f'K-means Clustering (Iteration {i+1})')
    plt.show()

    distances = np.sqrt(((X - centroids[cluster_assignments])**2).sum(axis=1))
    objective_function = np.sum(distances**2)
    print(f'Iteration {i+1}: K-means Objective Function Value: {objective_function:.2f}')

################################################################

import numpy as np
from sklearn.mixture import GaussianMixture

# Use GMM to cluster the training data
gmm = GaussianMixture(n_components=2, random_state=0).fit(X)

# Plot for training points and mean vectors
cluster_assignments = gmm.predict(X)
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=cluster_assignments)
plt.scatter(gmm.means_[:, 0], gmm.means_[:, 1], c='r', marker='x')
plt.title('Gaussian Mixture Model Clustering (k=2)')
plt.show()

# Print learned parameters
print("Learned Parameters:")
print(f"Means: \n{gmm.means_}")
print(f"Covariances: \n{gmm.covariances_}")
print(f"Weights: {gmm.weights_}")

# Calculate responsibility values of (1.5, 1.5)
def gaussian(x, mean, cov):
    diff = x - mean
    expo = np.exp(-0.5 * np.dot(np.dot(diff, np.linalg.inv(cov)), diff.T))
    return (1 / (np.sqrt(2 * np.pi * np.linalg.det(cov)))) * expo

new_point = np.array([1.5, 1.5])
responsibilities = []
for i in range(gmm.n_components):
    mean = gmm.means_[i]
    cov = gmm.covariances_[i]
    responsibilities.append(gmm.weights_[i] * gaussian(new_point, mean, cov))

# Print the responsibility vals
print(f"\nResponsibility values for (1.5, 1.5):")
print(responsibilities)

# Assign the new point to a cluster
assignment_by_responsibility = np.argmax(responsibilities)
assignment_by_predict = gmm.predict([[1.5, 1.5]])[0]

print(f"\nCluster assignment for (1.5, 1.5):")
print(f"By responsibility: {assignment_by_responsibility}")
print(f"By predict function: {assignment_by_predict}")