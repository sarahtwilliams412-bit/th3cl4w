"""
Visual Hull Reconstruction — Agent 3

Takes synchronized ASCII frame pairs and produces a 128^3 3D occupancy
grid by intersecting the silhouette information from both views.

Core insight: with two orthogonal views (top-down for XY, profile for XZ),
occupancy(x, y, z) = min(top_density(x, y), prof_density(x, z))
— a single NumPy broadcast operation.
"""
