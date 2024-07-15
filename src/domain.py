from numpy import array, full_like, hstack, argsort
from numpy import int32 as npint32
from mpi4py import MPI
from dolfinx import mesh as doflinx_mesh

import utils
from constants import BoundaryFlag


class Domain(object):

    def __init__(
        self,
        length:float = None, 
        width:float = None,
        discretization_shape = None
    ):
        # Corners of the box
        lower_left_corner = array([0,0,0])
        upper_right_corner = array([length, width, width])

        # Dimensions
        self.length = length
        self.width = width

        # 
        self.mesh = doflinx_mesh.create_box(
            MPI.COMM_WORLD,
            [
                lower_left_corner, 
                upper_right_corner
            ],
            list(discretization_shape),
            cell_type=doflinx_mesh.CellType.hexahedron
        )
        self.shape = tuple([x+1 for x in discretization_shape])
        
        # Dimensions
        self.ambient_dim = self.mesh.topology.dim
        self.facet_dim = self.mesh.topology.dim - 1
        self.edge_dim = self.mesh.topology.dim - 2

        # Boundaries
        boundaries = [
            utils.get_locator(BoundaryFlag.TOP, upper_right_corner),
            utils.get_locator(BoundaryFlag.RIGHT, upper_right_corner),
            utils.get_locator(BoundaryFlag.FRONT, upper_right_corner),
            utils.get_locator(BoundaryFlag.BOTTOM, lower_left_corner),
            utils.get_locator(BoundaryFlag.LEFT, lower_left_corner),
            utils.get_locator(BoundaryFlag.BACK, lower_left_corner)
        ]

        facet_indices, facet_markers = [], []
        for marker, locator in boundaries:
            facets = doflinx_mesh.locate_entities_boundary(
                self.mesh, 
                self.facet_dim,
                locator
            )
            facet_indices.append(facets)
            facet_markers.append(full_like(facets, int(marker)))

        facet_indices = hstack(facet_indices).astype(npint32)
        facet_markers = hstack(facet_markers).astype(npint32)
        sorted_facets = argsort(facet_indices)
        facet_tag = doflinx_mesh.meshtags(
            self.mesh,
            self.facet_dim,
            facet_indices[sorted_facets],
            facet_markers[sorted_facets]
        )