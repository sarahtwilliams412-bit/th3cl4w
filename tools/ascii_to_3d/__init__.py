"""ASCII-to-3D converter.

Takes two ASCII art images representing orthographic projections of a physical
object (front view and side view) and reconstructs a three-dimensional virtual
object using visual-hull / shape-from-silhouette techniques.

Output formats:
  - Wavefront OBJ mesh file (loadable in Blender, MeshLab, etc.)
  - 3D ASCII art preview rendered to the terminal
"""
