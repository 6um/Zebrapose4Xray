import numpy as np
import pyvista as pv
from sklearn.cluster import KMeans

file = "./welsh-dragon-small-centered.stl"
mesh = pv.read(file)
vertices = np.array(mesh.points)

visualize = False
max_level = 10

from k_means_constrained import KMeansConstrained

def balanced_kmeans_split(X, random_state=42):
    n = len(X)

    if n < 2:
        return np.zeros(n, dtype=int)

    half = n // 2

    clf = KMeansConstrained(
        n_clusters=2,
        size_min=half,
        size_max=n - half,
        random_state=random_state
    )

    labels = clf.fit_predict(X)
    return labels

# Level 1
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels_prev = kmeans.fit_predict(vertices)
# labels_prev = balanced_kmeans_split(vertices)

mesh_level_1 = pv.read(file)
mesh_level_1["clusters"] = labels_prev
mesh_level_1.save("binary_level_1.vtk")

print("Level 1 - 2 clusters:")
for i in range(2):
    print(f"  Cluster {i}: {np.sum(labels_prev == i)} vertices")

if visualize:
    plotter = pv.Plotter()
    plotter.add_mesh(mesh_level_1, scalars="clusters", cmap="Greys", show_edges=False, show_scalar_bar=False)
    plotter.add_axes()
    plotter.show()

# Level 2 ~ max_level
for level in range(2, max_level + 1):
    num_prev_clusters = 2 ** (level - 1)
    num_clusters = 2 ** level
    labels_current = np.zeros(len(vertices), dtype=int)

    # for cluster_id in range(num_prev_clusters):
    #     mask = labels_prev == cluster_id
    #     vertices_cluster = vertices[mask]
    #
    #     if len(vertices_cluster) >= 2:
    #         sub_labels = balanced_kmeans_split(vertices_cluster)
    #         labels_current[mask] = sub_labels + cluster_id * 2
    #     else:
    #         labels_current[mask] = cluster_id * 2

    for cluster_id in range(num_prev_clusters):
        mask = labels_prev == cluster_id
        vertices_cluster = vertices[mask]

        if len(vertices_cluster) >= 2:
            kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
            sub_labels = kmeans.fit_predict(vertices_cluster)
            labels_current[mask] = sub_labels + cluster_id * 2
        elif len(vertices_cluster) == 1:
            labels_current[mask] = cluster_id * 2

    mesh_current = pv.read(file)
    mesh_current["clusters"] = labels_current
    mesh_current.save(f"binary_level_{level}.vtk")

    print(f"Level {level} - {num_clusters} clusters:")
    for i in range(num_clusters):
        count = np.sum(labels_current == i)
        if count > 0:
            print(f"  Cluster {i}: {count} vertices")

    if visualize:
        plotter = pv.Plotter()
        plotter.add_mesh(mesh_current, scalars="clusters", cmap="viridis", show_edges=False, show_scalar_bar=False)
        plotter.add_axes()
        plotter.show()

    labels_prev = labels_current.copy()

print(f"\nClustering complete up to level {max_level}!")