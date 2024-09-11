from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# Load the data
lib = np.load("/")
data = lib["data"]
labels = lib["labels"]

# Compute PCA
data_means = np.mean(data, axis=0)
norm_data = data - data_means
_, _, Vh = np.linalg.svd(norm_data)
pca_data = np.matmul(norm_data, Vh[:3].T)

# Create a 3D scatter plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Define the axes
x = pca_data[:, 0]  # PCA dimension 1
y = pca_data[:, 1]  # PCA dimension 2
z = pca_data[:, 2]  # PCA dimension 3

# Plot the scatter plot
scatter = ax.scatter(x, y, z, c=labels, cmap='plasma', marker='o')

# Add labels and title
ax.set_xlabel('U1')
ax.set_ylabel('U2')
ax.set_zlabel('U3')
ax.set_title('PCA of Iris Dataset')

# Add color bar
cbar = plt.colorbar(scatter)
cbar.set_label('Species')
cbar.set_ticks([0, 1, 2])
cbar.set_ticklabels(['Iris Setosa', 'Iris Versicolor', 'Iris Virginica'])

# Show the plot
plt.show()
