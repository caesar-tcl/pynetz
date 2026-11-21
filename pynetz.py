import numpy as np
from scipy.spatial.distance import pdist, squareform


# # # rough estimate of maximum distance for a sphere # # #
number_of_points = 500
surface_area = 4 * np.pi
maximum_distance = np.sqrt(2 * surface_area / (np.sqrt(3) * number_of_points))


def pyNetz(coordinates, cutoff = 2 * maximum_distance, tetrahedral_angle = 5/6*np.pi, flip_initial_triangle = True):
    """
    Triangulate a point cloud using an advancing-front approach.

    Parameters:
    - coordinates: np.ndarray
        Array of point coordinates (N x 3).
    - cutoff: float
        Maximum distance for edges to be considered valid.
    - tetrahedral_angle: float
        Maximum angle between adjacent triangle normals.
    - flip_initial_triangle: bool
        Reverse the initial triangle orientation.

    Returns:
    - valid_triangles: np.ndarray
        Array of triangles (each row contains three vertex indices).
    - valid_edges: np.ndarray
        Array of all edges below the cutoff.
    - valid_edge_lengths: np.ndarray
        Lengths corresponding to valid_edges.
    - normal_vectors: np.ndarray
        Normal vectors of all triangles.
    - adjacent_triangles: list
        Indices of adjacent triangles for each triangle.
    - boundary_edges: np.ndarray
        Edges belonging to exactly one triangle.
    """

    # # # Initialization # # #

    n = len(coordinates)

    distances = pdist(coordinates)
    distance_matrix = squareform(distances)

    edge_i, edge_j = np.triu_indices(n, 1)
    valid = distances < cutoff

    valid_edge_lengths = distances[valid]
    valid_edges = np.column_stack((edge_i[valid], edge_j[valid]))

    empty_triangles = np.empty((0, 3), dtype=int)
    empty_edges = np.empty((0, 2), dtype=int)
    empty_normals = np.empty((0, 3))

    if len(valid_edges) == 0:
        return empty_triangles, empty_edges, np.empty(0), empty_normals, [], empty_edges.copy()

    adjacency = np.zeros((n, n), dtype=bool)
    adjacency[valid_edges[:, 0], valid_edges[:, 1]] = True
    adjacency[valid_edges[:, 1], valid_edges[:, 0]] = True

    edge_count = np.zeros((n, n), dtype=np.uint8)
    edge_triangle = np.full((n, n), -1, dtype=int)
    edge_triangle_neighbor = np.full((n, n), -1, dtype=int)

    # # # Initial Triangle # # #

    a, b = valid_edges[np.argmin(valid_edge_lengths)]

    candidates = np.flatnonzero(adjacency[a] & adjacency[b])

    if len(candidates) == 0:
        return empty_triangles, valid_edges, valid_edge_lengths, empty_normals, [], empty_edges.copy()

    perimeters = distance_matrix[a, b] + distance_matrix[a, candidates] + distance_matrix[b, candidates]
    c = candidates[np.argmin(perimeters)]

    triangles = [[a, b, c]]

    for u, v in ((a, b), (b, c), (c, a)):
        u, v = min(u, v), max(u, v)
        edge_count[u, v] = edge_count[v, u] = 1
        edge_triangle[u, v] = edge_triangle[v, u] = 0

    boundary = [(min(a, b), max(a, b)), (min(b, c), max(b, c)), (min(c, a), max(c, a))]

    # # # Triangulation # # #

    idx = 0

    while idx < len(boundary):

        a, b = boundary[idx]
        idx += 1

        if edge_count[a, b] != 1:
            continue

        adjacent_triangle = triangles[edge_triangle[a, b]]
        avoid_vertex = sum(adjacent_triangle) - a - b

        candidates = np.flatnonzero(adjacency[a] & adjacency[b])
        candidates = candidates[
            (candidates != avoid_vertex) &
            (edge_count[a, candidates] < 2) &
            (edge_count[b, candidates] < 2)
        ]

        if len(candidates) == 0:
            continue

        edge_vector = coordinates[a] - coordinates[b]
        reference_normal = np.cross(edge_vector, coordinates[avoid_vertex] - coordinates[b])
        reference_norm = np.linalg.norm(reference_normal)

        if reference_norm == 0:
            continue

        candidate_normals = np.cross(edge_vector, coordinates[candidates] - coordinates[b])
        candidate_norms = np.linalg.norm(candidate_normals, axis=1)

        angles = np.full(len(candidates), np.inf)
        nondegenerate = candidate_norms > 0

        angles[nondegenerate] = np.arccos(np.clip(
            -(candidate_normals[nondegenerate] @ reference_normal) /
            (reference_norm * candidate_norms[nondegenerate]),
            -1, 1
        ))

        candidates = candidates[angles < tetrahedral_angle]

        if len(candidates) == 0:
            continue

        perimeters = distance_matrix[a, b] + distance_matrix[a, candidates] + distance_matrix[b, candidates]
        vertex = candidates[np.argmin(perimeters)]

        # orient the new triangle consistently with its neighbor
        same_direction = any(
            adjacent_triangle[k] == a and adjacent_triangle[(k + 1) % 3] == b
            for k in range(3)
        )

        triangle = [b, a, vertex] if same_direction else [a, b, vertex]
        triangles.append(triangle)

        new_index = len(triangles) - 1

        # update edge information and boundary
        for u, v in ((triangle[0], triangle[1]), (triangle[1], triangle[2]), (triangle[2], triangle[0])):
            u, v = min(u, v), max(u, v)

            if edge_count[u, v] == 0:
                edge_count[u, v] = edge_count[v, u] = 1
                edge_triangle[u, v] = edge_triangle[v, u] = new_index
                boundary.append((u, v))

            elif edge_count[u, v] == 1:
                edge_count[u, v] = edge_count[v, u] = 2
                edge_triangle_neighbor[u, v] = edge_triangle_neighbor[v, u] = new_index

    # # # Final Orientation # # #

    valid_triangles = np.asarray(triangles, dtype=int)

    if flip_initial_triangle:
        valid_triangles[:, [1, 2]] = valid_triangles[:, [2, 1]]

    # calculate normal vectors
    v1 = coordinates[valid_triangles[:, 1]] - coordinates[valid_triangles[:, 0]]
    v2 = coordinates[valid_triangles[:, 2]] - coordinates[valid_triangles[:, 0]]
    normal_vectors = np.cross(v1, v2)

    # find adjacent triangles
    adjacent_triangles = [[] for _ in valid_triangles]

    for u, v in np.argwhere(np.triu(edge_count == 2, 1)):
        triangle_1 = edge_triangle[u, v]
        triangle_2 = edge_triangle_neighbor[u, v]

        adjacent_triangles[triangle_1].append(triangle_2)
        adjacent_triangles[triangle_2].append(triangle_1)

    adjacent_triangles = [np.asarray(t, dtype=int) for t in adjacent_triangles]

    # find boundary edges
    boundary_edges = np.argwhere(np.triu(edge_count == 1, 1))

    return valid_triangles, valid_edges, valid_edge_lengths, normal_vectors, adjacent_triangles, boundary_edges

