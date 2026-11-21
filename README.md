<p align="center">
  <img src="pynetz.png" alt="PyNetz Logo" width="400">
</p>

`pynetz.py` contains PyNetz, a custom lightweight, CGAL-inspired Python mesh triangulator originally developed in 2023 for my master’s thesis on cell growth.

Just supply an unordered 3D point cloud, and the code will generate:

* **`valid_triangles`**: Triangulated faces.
* **`valid_edges`**: All edges shorter than the cutoff.
* **`valid_edge_lengths`**: Lengths corresponding to `valid_edges`.
* **`normal_vectors`**: Normal vector of each triangle, in the same order as `valid_triangles`.
* **`adjacent_triangles`**: Indices of neighboring triangles for each triangle.
* **`boundary_edges`**: Edges belonging to exactly one triangle in the resulting mesh.

Because it uses a greedy, brute-force algorithm, it may occasionally discard points that violate geometric constraints, meaning the number of unique points in the final mesh can be smaller than the original input, `len(unique_triangle_points) != len(original_points)`.
