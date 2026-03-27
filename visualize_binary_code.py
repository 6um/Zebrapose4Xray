# Visualize Level n - highlight only a single cluster

import trimesh
import pyvista as pv
level = 8
mesh_level_n = pv.read(f"binary_level_{level}.vtk")


# Create binary mask: 1 for selected cluster, 0 for all others
binary = (mesh_level_n['clusters'] % 2).astype(int)
mesh_level_n['clusters'] = binary

plotter = pv.Plotter()
plotter.add_mesh(mesh_level_n, scalars='clusters', cmap='Greys', show_edges=False, show_scalar_bar=False)
plotter.add_axes()
plotter.show()

print(f"Level {level} visualization - binary cluster")



