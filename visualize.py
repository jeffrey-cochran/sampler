
import pyvista
from dolfinx import plot

def visualize_deformation(
        *,
        domain=None,
        displacements=None,
        function_space=None,
        file_name=None,
        exaggeration_factor=None
):
    plotter = pyvista.Plotter(off_screen=True)
    topology, cell_types, geometry = plot.vtk_mesh(function_space)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

    # Attach vector values to grid and warp grid by vector
    grid["u"] = displacements.x.array.reshape((geometry.shape[0], 3))
    warped = grid.warp_by_vector("u", factor=exaggeration_factor)

    # Add original and deformed meshes to plotter
    actor_0 = plotter.add_mesh(grid, style="wireframe", color="k")
    actor_1 = plotter.add_mesh(warped, show_edges=True)

    plotter.show_axes()
    plotter.screenshot(f"{file_name}.jpg")