import geometry

A = geometry.UnitSquare(10)

print(A.boundary_idx(A.BoundaryFlag.RIGHT, A.BoundaryFlag.BOTTOM))